import math
import time
import sys
from typing import Iterable
import torch
import torch.nn as nn
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from utilities import *
import os
from dataloader import rand_mask_generate, mask_expand2d
from torch.cuda.amp import autocast,GradScaler
import datetime

def contrastive_loss(audio_rep, video_rep, bidirect_contrast=False):
    # calculate nce loss for mean-visual representation and mean-audio representation          
    audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
    video_rep = torch.nn.functional.normalize(video_rep, dim=-1)       
    total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05       
    # by default we use single directional
    if bidirect_contrast == False:
        nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
        return nce, c_acc
    else:
        nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
        c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
        c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
        nce = (nce_1 + nce_2) / 2
        c_acc = (c_acc_1 + c_acc_2) / 2
        return nce, c_acc    


def attn_mask_generator(attn, mask_ratio, num_frames):
    """
    Generate a boolean mask for attention based on the attention scores and mask ratio.
    Args:
        attn (torch.Tensor): Attention scores of shape (B*T, N).
        mask_ratio (float): Ratio of the attention to be masked.
        num_frames (int): Number of frames in the video.
    Returns:
        bool_masked_pos (torch.Tensor): Boolean mask of shape (B, N) indicating which positions are masked.
    """
    # get attn logits:
    attn = attn.softmax(dim=-1)  # Apply softmax to get probabilities

    BT, N = attn.shape
    T = num_frames
    B = BT // T
    bool_masked_pos = torch.ones((BT, N), dtype=torch.bool, device=attn.device)
    num_mask = int(N * mask_ratio)
    N_vis = N - num_mask
    pos_1 = torch.arange(BT, device=attn.device).view(-1, 1).repeat(1, N_vis)
    importance = torch.multinomial(attn, N)
    pos_2 = importance[:, :N_vis]
    # print("pos_1 shape: ", pos_1.shape, "pos_2 shape: ", pos_2.shape)
    bool_masked_pos[pos_1, pos_2] = 0
    bool_masked_pos = bool_masked_pos.view(B, -1)
    return bool_masked_pos


def resume_training(model, optimizer, exp_dir, device):
    """
    Resume training from the latest checkpoint if available.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        exp_dir (str): The directory where model checkpoints are saved.
        device (torch.device): The device to load the model and optimizer onto.

    Returns:
        int: The starting epoch for training.
    """
    checkpoint_path = os.path.join(exp_dir, "models", "best_model.pth")
    optimizer_path = os.path.join(exp_dir, "models", "best_optim_state.pth")
    if os.path.exists(checkpoint_path) and os.path.exists(optimizer_path):    
        # Load optimizer state
        optimizer_state = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(optimizer_state)
        # Extract the last saved epoch
        ckpt_list = os.listdir(os.path.join(exp_dir, "models"))
        epochs = [int(f.split('.')[-2]) for f in ckpt_list if f.endswith('.pth') and f.startswith('model')]
        start_epoch = max(epochs) if epochs else 0
        # # Load model weights
        checkpoint_path = os.path.join(exp_dir, "models", f"model.{start_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 1

    return start_epoch


def train_mamba(model, teacher_v, teacher_a, train_loader, test_loader, args, local_rank):
    """
    engine for training mamba model
    args contains the following parameters:
        contrastive_loss_weight (float): Weight for the contrastive loss.
        mask_ratio (float): Ratio of the input to be masked. 
    """
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    model.train()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    teacher_v = teacher_v.to(device)
    teacher_a = teacher_a.to(device)
    teacher_v.eval()
    teacher_a.eval()
    teacher_v = DDP(teacher_v, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    teacher_a = DDP(teacher_a, device_ids=[local_rank], output_device= local_rank, find_unused_parameters=True)
    
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    model = model.to(device)
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))
    if args.resume:
        epoch = resume_training(model, optimizer=optimizer, exp_dir=args.exp_dir, device=device)
        global_step = (epoch - 1) * len(train_loader) 
    epoch += 1
    scaler = GradScaler()
    
    loss_distillation = nn.MSELoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    while epoch < args.n_epochs + 1:
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        for i, (a_input, v_input, _, mask, mask_v, mask_a) in enumerate(train_loader):
            #print("input shape: ", a_input.shape, v_input.shape, mask.shape, mask_v.shape, mask_a.shape)
            #torch.Size([10, 1024, 64]) torch.Size([10, 16, 3, 224, 224]) torch.Size([10, 16, 216]) torch.Size([10, 16, 196]) torch.Size([10, 16, 4])
            B, T, C, H, W = v_input.shape
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()
            
            

            # Teacher Model Forward Pass - Optimized for memory efficiency
            with torch.no_grad():
                # Pre-create reusable tensors
                is_longer_tensor = torch.tensor([True], dtype=torch.bool, device=device)
                output_attentions_tensor = torch.tensor([True], dtype=torch.bool, device=device)
                
                # Clip pipeline and visual mask
                v_input = v_input.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
                clip_target, clip_attn = teacher_v(v_input) # K, B, 1961, 768
                mask_v = attn_mask_generator(clip_attn, args.mask_ratio, T)  # B, 1960
                C_CLIP = clip_target.shape[-1]  # 512
                mask_v = mask_v.reshape(B, T, -1)
                
                # Clear intermediate variables to save memory
                del clip_attn
                torch.cuda.empty_cache()
                
                # Clap pipeline and audio mask
                clap_input = a_input.unsqueeze(1)  # B, 1, 1024, 64
                audio_outputs = teacher_a(clap_input, is_longer=is_longer_tensor, output_attentions=output_attentions_tensor)
                clap_target, clap_attn = audio_outputs.last_hidden_state, audio_outputs.attentions[-1]
                
                # Clear audio outputs to save memory
                del audio_outputs
                _, C_CLAP, num_freq_bins, _ = clap_target.shape  # B, 768, 2, 32
                # Optimize tensor operations - use in-place operations where possible
                clap_target = rearrange(clap_target, 'b d freq_bins (t time_bins) -> b d t freq_bins time_bins', t=T)
                clap_target = clap_target.flatten(start_dim=2).permute(0, 2, 1)  # B, 64, 768
                
                # Clear attention weights to save memory
                del clap_attn
                torch.cuda.empty_cache()
                
                # Optimize mask operations - reduce intermediate tensor creation
                mask_a_ori = mask_a.clone() # B, T, 4
                # More efficient mask expansion
                mask_a = mask_a.view(B, T, 2, 2)
                mask_a = mask_a.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                mask_a = mask_a.view(B, T, -1)  # B, 16, 16

                mask_v = mask_v.bool().to(device)
                mask_a = mask_a.bool().to(device)
                zeros = torch.zeros(B, T, 1).bool().to(device)  # B, T, 1
                mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=2)  # B, T, 196 + 16 + 4
                mask = mask.reshape(B, -1)
                # get target features
                mask_v = mask_v.reshape(B, -1)
                mask_v = torch.cat((torch.ones(B, 1).to(device), mask_v), dim=1).bool()  # B, 1961  # the cls token wont join the loss calculation
                clip_target = clip_target.squeeze(0) if len(clip_target.shape) == 4 else clip_target
                clip_target = clip_target[~mask_v].reshape(B, -1, C_CLIP)

                mask_a = mask_a_ori.reshape(B, -1)
                clap_target = clap_target[~mask_a].reshape(B, -1, C_CLAP)
                # Normalize the targets
                clip_target = torch.nn.functional.normalize(clip_target, dim=-1)
                clap_target = torch.nn.functional.normalize(clap_target, dim=-1)

            with autocast():
                a_input = a_input.permute(0, 2, 1)  # B, C, T
                #print("input shape: ", a_input.shape, v_input.shape, mask.shape)
                x_clip, x_clap, global_v, global_a = model(v_input, a_input, mask)
                pred_clap = x_clap[:, :, 3::4, :]               # use every 4th state as the CLAP prediction
                # global_v = global_v.mean(dim=1)  # B, 16
                # global_a = global_a.mean(dim=1)  # B, 16
                B, T, D = global_v.shape
                global_v = global_v.reshape(B, -1)  # B, 16* D
                global_a = global_a.reshape(B, -1)  # B, 16 * D
                # print("global_v shape: ", global_v.shape, "global_a shape
                loss_c, c_acc = contrastive_loss(global_v, global_a)
                loss_c = loss_c * args.contrastive_loss_weight
                loss_a = loss_distillation(pred_clap, clap_target)* 10
                loss_v = loss_distillation(x_clip, clip_target)* 10

                loss = loss_a + loss_v +  loss_c
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_a_meter.update(loss_a.item(), B)
            loss_v_meter.update(loss_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            loss_av_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            # log:
            if args.use_wandb and local_rank == 0:
                wandb.log({
                    'train vision loss': loss_v.item(),
                    'train audio loss': loss_a.item(),
                    'train contra loss': loss_c.item(),
                    'train loss_all': loss.item(),
                    'iters': (epoch - 1) * len(train_loader) + i,
                    'epoch': epoch
                })
            args.n_print_steps = 100
            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step
            # print_step = True # for debug
            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Total Loss {loss_av_meter.val:.4f}\t'
                  'Train T-S Loss Audio {loss_a_meter.val:.4f}\t'
                  'Train T-S Loss Visual {loss_v_meter.val:.4f}\t'
                  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                  'Train Contrastive Acc {c_acc:.3f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        eval_loss_all, eval_loss_v, eval_loss_a, eval_loss_c = validate_mamba(model=model, teacher_v=teacher_v, teacher_a=teacher_a, test_loader=test_loader, args=args, local_rank=local_rank)
 
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_v))
        print("Eval Total Loss: {:.6f}".format(eval_loss_all))
        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        eval_loss_av = eval_loss_all
        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if not os.path.exists("%s/models" % (exp_dir)):
            os.makedirs("%s/models" % (exp_dir))
        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()


def validate_mamba(model, teacher_v, teacher_a, test_loader, args, local_rank=0):
    """
    Validate the Mamba model on the test dataset.
    Args:
        model (torch.nn.Module): The Mamba model to be validated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        args: Arguments containing configuration parameters.
        local_rank (int): Local rank for distributed training.
    Returns:
        tuple: Average loss and accuracy metrics.
    """
    device = torch.device(f'cuda:{local_rank}')
    model.eval()
    model = model.to(device)
    
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    c_acc_meter = AverageMeter()
    loss_distillation = nn.MSELoss()
    with torch.no_grad():
        for i, (a_input, v_input, _, mask, mask_v, mask_a) in enumerate(test_loader):
            if i> 10:
                break
            B, T, C, H, W = v_input.shape
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            is_longer_tensor = torch.tensor([True], dtype=torch.bool, device=device)
            output_attentions_tensor = torch.tensor([True], dtype=torch.bool, device=device)
            
            # Clip pipeline and visual mask
            v_input = v_input.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
            clip_target, clip_attn = teacher_v(v_input) # K, B, 1961, 768
            mask_v = attn_mask_generator(clip_attn, args.mask_ratio, T)  # B, 1960
            C_CLIP = clip_target.shape[-1]  # 512
            mask_v = mask_v.reshape(B, T, -1)
            
            # Clear intermediate variables to save memory
            del clip_attn
            torch.cuda.empty_cache()
            
            # Clap pipeline and audio mask
            clap_input = a_input.unsqueeze(1)  # B, 1, 1024, 64
            audio_outputs = teacher_a(clap_input, is_longer=is_longer_tensor, output_attentions=output_attentions_tensor)
            clap_target, clap_attn = audio_outputs.last_hidden_state, audio_outputs.attentions[-1]
            
            # Clear audio outputs to save memory
            del audio_outputs
            _, C_CLAP, num_freq_bins, _ = clap_target.shape  # B, 768, 2, 32
            # Optimize tensor operations - use in-place operations where possible
            clap_target = rearrange(clap_target, 'b d freq_bins (t time_bins) -> b d t freq_bins time_bins', t=T)
            clap_target = clap_target.flatten(start_dim=2).permute(0, 2, 1)  # B, 64, 768
            
            # Clear attention weights to save memory
            del clap_attn
            torch.cuda.empty_cache()
            
            # Optimize mask operations - reduce intermediate tensor creation
            mask_a_ori = mask_a.clone() # B, T, 4
            # More efficient mask expansion
            mask_a = mask_a.view(B, T, 2, 2)
            mask_a = mask_a.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            mask_a = mask_a.view(B, T, -1)  # B, 16, 16

            mask_v = mask_v.bool().to(device)
            mask_a = mask_a.bool().to(device)
            zeros = torch.zeros(B, T, 1).bool().to(device)  # B, T, 1
            mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=2)  # B, T, 196 + 16 + 4
            mask = mask.reshape(B, -1)
            # get target features
            mask_v = mask_v.reshape(B, -1)
            mask_v = torch.cat((torch.ones(B, 1).to(device), mask_v), dim=1).bool()  # B, 1961  # the cls token wont join the loss calculation
            clip_target = clip_target.squeeze(0) if len(clip_target.shape) == 4 else clip_target
            clip_target = clip_target[~mask_v].reshape(B, -1, C_CLIP)

            mask_a = mask_a_ori.reshape(B, -1)
            clap_target = clap_target[~mask_a].reshape(B, -1, C_CLAP)
            # Normalize the targets
            clip_target = torch.nn.functional.normalize(clip_target, dim=-1)
            clap_target = torch.nn.functional.normalize(clap_target, dim=-1)

            a_input = a_input.permute(0, 2, 1)  # B, C, T
            #print("input shape: ", a_input.shape, v_input.shape, mask.shape)
            x_clip, x_clap, global_v, global_a = model(v_input, a_input, mask)
            pred_clap = x_clap[:, :, 3::4, :]               # use every 4th state as the CLAP prediction
            # global_v = global_v.mean(dim=1)  # B, 16
            # global_a = global_a.mean(dim=1)  # B, 16
            B, T, D = global_v.shape
            global_v = global_v.reshape(B, -1)  # B, 16* D
            global_a = global_a.reshape(B, -1)  # B, 16 * D
            loss_c, c_acc = contrastive_loss(global_v, global_a)
            loss_c = loss_c * args.contrastive_loss_weight
            loss_a = loss_distillation(pred_clap, clap_target)* 10
            loss_v = loss_distillation(x_clip, clip_target)* 10
            loss = loss_a + loss_v +  loss_c
            loss_a_meter.update(loss_a.item(), B)
            loss_v_meter.update(loss_v.item(), B)   
            loss_c_meter.update(loss_c.item(), B)
            loss_av_meter.update(loss.item(), B)
        return loss_av_meter.avg, loss_v_meter.avg, loss_a_meter.avg, loss_c_meter.avg
        