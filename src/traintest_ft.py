# -*- coding: utf-8 -*-
# Mod by Linge Wang

# Originally from:
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from sklearn.metrics import average_precision_score

def resume_training_ft(audio_model, optimizer, exp_dir, device):
    """
    Resume training from the latest checkpoint if available.

    Args:
        audio_model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        exp_dir (str): The directory where model checkpoints are saved.
        device (torch.device): The device to load the model and optimizer onto.

    Returns:
        int: The starting epoch for training.
    """
    checkpoint_path = os.path.join(exp_dir, "models", "best_audio_model.pth")
    optimizer_path = os.path.join(exp_dir, "models", "best_optim_state.pth")

    if os.path.exists(checkpoint_path) and os.path.exists(optimizer_path):
        
        
        # Load optimizer state
        optimizer_state = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(optimizer_state)
        # Extract the last saved epoch
        ckpt_list = os.listdir(os.path.join(exp_dir, "models"))
        epochs = [int(f.split('.')[-2]) for f in ckpt_list if f.endswith('.pth') and f.startswith('audio_model')]
        start_epoch = max(epochs) if epochs else 0
        # # Load model weights
        checkpoint_path = os.path.join(exp_dir, "models", f"audio_model.{start_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        audio_model.load_state_dict(checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 1

    return start_epoch


def calculate_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def train(audio_model, train_loader, test_loader, args, local_rank):
    device = torch.device(f'cuda:{local_rank}')
    audio_model.to(device)
    if args.use_dist:
        audio_model = DDP(audio_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    best_acc = 0
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir


    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
                'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
                'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
                'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if args.freeze_base == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # Learning rate scheduler selection strategy:
    # 1. ReduceLROnPlateau: Good for exploration and hyperparameter tuning
    #    - Pros: Adaptive, prevents overfitting, no need to preset decay epochs
    #    - Cons: Non-reproducible, computationally expensive, may be too conservative
    # 2. MultiStepLR: Better for final experiments and paper results
    #    - Pros: Reproducible, efficient, predictable, stable
    #    - Cons: Requires domain knowledge, less adaptive
    # 
    # Recommendation: Use ReduceLROnPlateau for initial experiments to find good decay points,
    # then switch to MultiStepLR with fixed milestones for final reproducible results
    
    # only for preliminary test, formal exps should use fixed learning rate scheduler
    if args.lr_adapt == True:
        # Adaptive scheduler: good for exploration but less reproducible
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        # Fixed scheduler: better for reproducible final results
        # Consider using decay points found from ReduceLROnPlateau experiments
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    main_metrics = args.metrics
    if args.loss == 'BCE':      # For multilabel classification
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':     # For Single-label classification
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    if args.resume:
        epoch = resume_training_ft(audio_model, optimizer=optimizer, exp_dir=args.exp_dir, device=device)
        global_step = (epoch - 1) * len(train_loader)
    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    args.n_print_steps = 10
    audio_model.train()
    while epoch < args.n_epochs + 1:
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        for i, (a_input, v_input, labels) in enumerate(train_loader):
            B = a_input.size(0)
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode)
                loss = loss_fn(audio_output, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            grad_norm = calculate_grad_norm(audio_model)
            if args.use_wandb and local_rank == 0:
                wandb.log({
                    'training loss': loss.item(),
                    'grad_norm':grad_norm,
                    'iters': (epoch - 1) * len(train_loader) + i,
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'mlp_lr': optimizer.param_groups[1]['lr'],
                })
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            #print_step = True
            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1


        print('start validation')

        if args.raw_data == 'k700':
            acc, valid_loss =  validate(audio_model, test_loader, args, local_rank=local_rank)
            print("acc: {:.6f}".format(acc))
            if acc > best_acc:
                    best_epoch = epoch
        else:
            mAP, valid_loss = validate(audio_model, test_loader, args, local_rank=local_rank, return_ap=True)
            print("mAP:{:.6f}".format(mAP))
            if mAP > best_mAP:
                best_mAP = mAP
                if main_metrics == 'mAP':
                    best_epoch = epoch
        # if main_metrics == 'mAP':
        #     mAP, valid_loss = validate(audio_model, test_loader, args, local_rank=local_rank, return_ap=True)
        #     print("mAP:{:.6f}".format(mAP))
        #     if mAP > best_mAP:
        #         best_epoch = epoch
        # stats, valid_loss = validate(audio_model, test_loader, args, local_rank)

        # mAP = np.mean([stat['AP'] for stat in stats])
        # mAUC = np.mean([stat['auc'] for stat in stats])
        # acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        # if main_metrics == 'mAP':
        #     print("mAP: {:.6f}".format(mAP))
        # else:
        #     print("acc: {:.6f}".format(acc))
        # print("AUC: {:.6f}".format(mAUC))
        # print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        print('validation finished')
        if args.use_wandb and local_rank == 0:
                wandb.log({
                    'valid_loss': valid_loss,
                    'epoch': epoch,
                    #'mAP': mAP if main_metrics == 'mAP' else None,
                    'acc': acc if main_metrics == 'acc' else None,
                })

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        save_interval = 5
        if args.use_video:
            save_interval = 1
        if args.save_model == True and (epoch % save_interval == 0 or epoch == args.n_epochs):
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
                
            elif main_metrics == 'acc':
                scheduler.step(acc)
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
        loss_meter.reset()
        per_sample_dnn_time.reset()


def validate(audio_model, val_loader, args, local_rank, output_pred=False, return_ap = False):
    device = torch.device(f'cuda:{local_rank}')
    batch_time = AverageMeter()
    # if not isinstance(audio_model, nn.DataParallel):
    #     audio_model = nn.DataParallel(audio_model)
    # audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode)
            predictions = audio_output.to('cpu').detach()
            predictions = torch.sigmoid(audio_output.float())
            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        if return_ap:
            metrics = calculate_mAP(audio_output, target)
        else:
            metrics = calculate_acc(audio_output, target, k=1)
        

        #stats = calculate_stats(audio_output, target)

    if output_pred == False:
        return metrics, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return metrics, audio_output, target
    
def calculate_acc(pred, target, k=1):
    """
    Calculate top-k accuracy.
    
    Args:
        pred: Tensor of shape (N, C) - model logits or probabilities
        target: Tensor of shape (N, C) - one-hot or multi-hot ground truth
        k: int, top-k value
        
    Returns:
        Top-k accuracy (float)
    """
    # Get top-k predicted class indices for each sample
    topk_pred = pred.topk(k, dim=1).indices  # shape: (N, k)

    # Expand target to indices (non-zero locations)
    target_indices = target.nonzero(as_tuple=False)  # shape: (num_positives, 2)
    target_dict = {i.item(): [] for i in target_indices[:, 0]}
    for i, j in target_indices:
        target_dict[i.item()].append(j.item())
    
    correct = 0
    for i in range(pred.size(0)):
        true_labels = target_dict.get(i, [])
        pred_labels = topk_pred[i].tolist()
        if any(label in pred_labels for label in true_labels):
            correct += 1

    return correct / pred.size(0)

def calculate_mAP(pred, target):
    """
    Calculate mean average precision (mAP) for multi-label classification.

    Args:
        pred (Tensor): shape (N, C), model outputs (logits or probabilities)
        target (Tensor): shape (N, C), one-hot or multi-hot ground truth (int or bool)

    Returns:
        float: mean average precision across all classes
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    target = (target > 0.5).astype(int)
    pred_binary = (pred > 0.5).astype(np.float32)
    recall = ((pred_binary * target).sum() / target.sum()).item()
    print("Recall on train batch:", recall)
    num_classes = pred.shape[1]
    APs = []

    for i in range(num_classes):
        if target[:, i].sum() == 0:
            continue  # 跳过没有正样本的类别
        ap = average_precision_score(target[:, i], pred[:, i])
        APs.append(ap)

    return sum(APs) / len(APs) if APs else 0.0