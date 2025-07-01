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
from traintest_ft import calculate_acc, calculate_mAP


def train_uni_mamba(model, teacher, train_loader, test_loader, args, local_rank, modality='audio'):
    """
    Engine for training the Uni-Mamba model.
    Args:
        model (torch.nn.Module): The Uni-Mamba model to be trained.
        teacher (torch.nn.Module): The teacher model for distillation.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        args: Arguments containing configuration parameters.
        local_rank (int): Local rank for distributed training.
    """
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    model.train()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    teacher = teacher.to(device)
    teacher.eval()
    teacher = DDP(teacher, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
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
    # if args.resume:
    #     epoch = resume_training(model, optimizer=optimizer, exp_dir=args.exp_dir, device=device)
    #     global_step = (epoch - 1) * len(train_loader) 
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
            T = 16
            B = a_input.shape[0]
            
            a_input = a_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()
            
        
            # Teacher Model Forward Pass - Optimized for memory efficiency
            with torch.no_grad():
                # Pre-create reusable tensors
                is_longer_tensor = torch.tensor([True], dtype=torch.bool, device=device)
                output_attentions_tensor = torch.tensor([True], dtype=torch.bool, device=device)
                # Clap pipeline and audio mask
                clap_input = a_input.unsqueeze(1)  # B, 1, 1024, 64
                audio_outputs = teacher(clap_input, is_longer=is_longer_tensor, output_attentions=output_attentions_tensor)
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

                mask_a = mask_a.bool().to(device)
                zeros = torch.zeros(B, T, 1).bool().to(device)  # B, T, 1
                mask = torch.cat([zeros, mask_a, zeros], dim=2)  # B, T, 16 + 4
                mask = mask.reshape(B, -1)
                # get target features
                mask_a = mask_a_ori.reshape(B, -1)
                clap_target = clap_target[~mask_a].reshape(B, -1, C_CLAP)
                # Normalize the targets
                clap_target = torch.nn.functional.normalize(clap_target, dim=-1)

            with autocast():
                a_input = a_input.permute(0, 2, 1)  # B, C, T

                outputs = model(a_input, mask)  # outputs is a dict
                x_clap = outputs['features']  # x_clap shape: K, B, 64, 512])

                pred_clap = x_clap[:, :, 3::4, :]               # use every 4th state as the CLAP prediction
                loss_a = loss_distillation(pred_clap[0], clap_target)* 10
                loss = loss_a 
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_a_meter.update(loss_a.item(), B)
            loss_av_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])
            
            # log:
            if args.use_wandb and local_rank == 0:
                wandb.log({
                    'train audio loss': loss_a.item(),
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
                  'Train T-S Loss Audio {loss_a_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time,  loss_a_meter=loss_a_meter, ), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        eval_loss_all = validate_unimamba(model=model, teacher=teacher, test_loader=test_loader, args=args, local_rank=local_rank)
 

        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_all))
        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))


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
        loss_a_meter.reset()

def validate_unimamba(model,  teacher, test_loader, args, local_rank=0, modality='audio'):
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
    
    loss_meter = AverageMeter()
    loss_distillation = nn.MSELoss()
    with torch.no_grad():
        for i, (a_input, v_input, _, mask, mask_v, mask_a) in enumerate(test_loader):
            if i> 10:
                break
            T = 16
            B = a_input.shape[0]
            
            a_input = a_input.to(device, non_blocking=True)
            is_longer_tensor = torch.tensor([True], dtype=torch.bool, device=device)
            output_attentions_tensor = torch.tensor([True], dtype=torch.bool, device=device)
            
            
            # Clap pipeline and audio mask
            clap_input = a_input.unsqueeze(1)  # B, 1, 1024, 64
            audio_outputs = teacher(clap_input, is_longer=is_longer_tensor, output_attentions=output_attentions_tensor)
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

            mask_a = mask_a.bool().to(device)
            zeros = torch.zeros(B, T, 1).bool().to(device)  # B, T, 1
            mask = torch.cat([zeros, mask_a, zeros], dim=2)  # B, T, 196 + 16 + 4
            mask = mask.reshape(B, -1)
            # get target features
            mask_a = mask_a_ori.reshape(B, -1)
            clap_target = clap_target[~mask_a].reshape(B, -1, C_CLAP)
            # Normalize the targets
            clap_target = torch.nn.functional.normalize(clap_target, dim=-1)

            a_input = a_input.permute(0, 2, 1)  # B, C, T
            #print("input shape: ", a_input.shape, v_input.shape, mask.shape)
            outputs = model(a_input, mask)  # outputs is a dict
            x_clap = outputs['features']  # x_clap shape: K, B, 64, 512])
            pred_clap = x_clap[:, :, 3::4, :]               # use every 4th state as the CLAP prediction
            loss_a = loss_distillation(pred_clap[0], clap_target)* 10
            loss = loss_a 
            loss_meter.update(loss.item(), B)
        return loss_meter.avg
    
def finetune_uni_mamba(model, train_loader, test_loader, args, local_rank):
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    model.train()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    torch.set_grad_enabled(True)
    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    # 线性探测区分 head / base 参数（如仅训练最后一层）
    probe_param_names = [k for k, v in model.named_parameters() if v.requires_grad and k.startswith('module.head')]
    base_param_names = [k for k, v in model.named_parameters() if v.requires_grad and not k.startswith('module.head')]
    probe_params = [v for k, v in model.named_parameters() if k in probe_param_names]
    base_params = [v for k, v in model.named_parameters() if k in base_param_names]
    print('Total parameter number is : {:.3f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Probe parameter number is : {:.3f} M'.format(sum(p.numel() for p in probe_params) / 1e6))  
    print('Base parameter number is : {:.3f} M'.format(sum(p.numel() for p in base_params) / 1e6))
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr},
        {'params': probe_params, 'lr': args.lr * args.head_lr}
    ], weight_decay=5e-7, betas=(0.95, 0.999))

    if args.lr_adapt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
            gamma=args.lrscheduler_decay)

    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    scaler = GradScaler()
    epoch += 1
    print("Start finetuning...")

    while epoch <= args.n_epochs:
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        begin_time = time.time()
        model.train()

        for i, (a_input, _, labels, _, _, _) in enumerate(train_loader):
            T = 16
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True).permute(0, 2, 1)  # B, C, T
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - begin_time)
            per_sample_data_time.update((time.time() - begin_time) / B)
            dnn_start_time = time.time()

            with autocast():
                logits = model(a_input)['logits']
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), B)
            per_sample_time.update((time.time() - begin_time) / B)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / B)

            if args.use_wandb and local_rank == 0:
                wandb.log({
                    'FT-training loss': loss.item(),
                    'iters': (epoch - 1) * len(train_loader) + i,
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr']
                })

            if global_step % args.n_print_steps == 0:
                print('Epoch: [{}/{}][{}/{}]\t Loss {:.4f}'.format(
                    epoch, args.n_epochs, i, len(train_loader), loss.item()))
                if np.isnan(loss_meter.avg):
                    print("training diverged.")
                    return
            global_step += 1
            begin_time = time.time()

        # ========== 验证 ==========
        print("Start validation...")
        if args.raw_data == 'k700':
            acc, valid_loss = validate_uni_ft(model, test_loader, args, local_rank)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
        else:
            mAP, valid_loss = validate_uni_ft(model, test_loader, args, local_rank, return_ap=True)
            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch

        if args.use_wandb and local_rank == 0:
            wandb.log({
                'valid_loss': valid_loss,
                'epoch': epoch,
                'acc': acc if args.metrics == 'acc' else None,
                'mAP': mAP if args.metrics == 'mAP' else None,
            })

        # Save best
        os.makedirs(f"{exp_dir}/models", exist_ok=True)
        if best_epoch == epoch:
            torch.save(model.state_dict(), f"{exp_dir}/models/best_model.pth")
            torch.save(optimizer.state_dict(), f"{exp_dir}/models/best_optim_state.pth")
        if args.save_model:
            torch.save(model.state_dict(), f"{exp_dir}/models/model.{epoch}.pth")
        save_interval = 1
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"{exp_dir}/models/model.{epoch}.pth")
            torch.save(optimizer.state_dict(), f"{exp_dir}/models/optim.{epoch}.pth")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if args.metrics == 'mAP':
                scheduler.step(mAP)
            else:
                scheduler.step(acc)
        else:
            scheduler.step()

        print(f'Epoch-{epoch} finished. LR: {optimizer.param_groups[0]["lr"]:.6f}')
        epoch += 1

        # reset meters
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()

def validate_uni_ft(model, val_loader, args, local_rank=0, return_ap=False):
    device = torch.device(f'cuda:{local_rank}')
    model.eval()
    batch_time = AverageMeter()
    A_predictions, A_targets, A_loss = [], [], []

    with torch.no_grad():
        for i, (a_input, _, labels, _, _, _) in enumerate(val_loader):
            if i > 10: break
            T = args.num_frames
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True).permute(0, 2, 1)
            labels = labels.to(device)

            with autocast():
                logits = model(a_input)['logits']
            predictions = torch.sigmoid(logits.float()).cpu()
            A_predictions.append(predictions)
            A_targets.append(labels.cpu())
            loss = args.loss_fn(logits, labels)
            A_loss.append(loss.cpu())

    predictions = torch.cat(A_predictions)
    targets = torch.cat(A_targets)
    loss = np.mean(A_loss)

    if return_ap:
        metric = calculate_mAP(predictions, targets)
    else:
        metric = calculate_acc(predictions, targets, k=1)

    return metric, loss
