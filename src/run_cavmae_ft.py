# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_ft import train, validate

# dist and log:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import socket


# finetune cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_val.json', help="training data json")
parser.add_argument("--data-val", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_val.json', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_class.csv', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=700, help="number of classes")
parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
parser.add_argument("--dataset", type=str, default="k700", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k700"])
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization", default=-5.081)
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization", default=4.4849)
parser.add_argument("--target_length", type=int, help="the input length in frames", default=1024)
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="exp/testmae06-bal-cav-mae-ft-5e-5-5-0.5-1-bs36-ldaFalse-multimodal-fzFalse-h100-a5", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=3, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval, default=False)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval, default=True)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default='/mnt/wanglinge/project/cav-mae/src/exp/testmae-audioset-cav-mae-balNone-lr5e-5-epoch25-bs256-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75/models/audio_model.25.pth', help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

parser.add_argument("--visible_gpus", type=str, default='0,1,2,3,4,5,6')
parser.add_argument("--wandb_project_name", type=str, default='cav')
parser.add_argument("--wandb_run_name", type=str, default='cav_finetune')
parser.add_argument("--use_wandb", action="store_true",
                        help="use wandb or not")
parser.add_argument("--wandb_id", type=str, default=None,
                        help="wandb id if resuming from a previous run")                        
parser.add_argument("--resume", action="store_true",
                        help="resume from a previous run")
parser.add_argument("--pooling", action="store_true",
                        help="use pooling or not")
parser.add_argument("--use_dist", default=True, type=bool, help="if use ddp" )
parser.add_argument("--raw_data", type=str, default="k700", help="raw data of daataset")

args = parser.parse_args()

if args.raw_data == 'as':
    args.modality = "audioonly"
else:
    args.modality = "both"
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def setup_distributed(visible_devices):
    #os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    setup_for_distributed(local_rank == 0)
    return local_rank

if args.use_dist:
    local_rank = setup_distributed(args.visible_gpus)
else:
    local_rank = 0

if args.use_wandb and local_rank == 0:
        print("init wandb")
        if args.wandb_id != None:
            args.resume = True
            print("resuming wandb run with id: ", args.wandb_id)
            wandb.init( project=args.wandb_project_name,
                entity='wanglg-institude-of-automation-cas',
                notes=socket.gethostname(),
                id=args.wandb_id,
                name=args.wandb_run_name,
                resume="must",
                job_type="training",
                reinit=True)
        else:
            wandb.init(project=args.wandb_project_name,
                   entity='wanglg-institude-of-automation-cas',
                   notes=socket.gethostname(),
                   name='cav_1',
                   job_type="training",
                   reinit=True)
        if args.wandb_run_name != None:
            wandb.run.name = args.wandb_run_name
        wandb.config.update(args)


# all exp in this work is based on 224 * 224 image
im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
              'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

if args.model == 'cav-mae-ft':
    print('finetune a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
    if args.raw_data == 'as':
        audio_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
    else:
        audio_model = models.CAVMAE_k700_FT(label_dim=args.n_class,pooling=args.pooling, modality_specific_depth=11)
    
    args.align = False
elif args.model == 'cav-mae-sync-ft':
    audio_model = models.CAVMAE_Sync_k700_FT(label_dim=args.n_class,pooling=args.pooling, audio_length=400,  modality_specific_depth=11)
    if args.raw_data != 'as':
        args.align = True
    else:
        args.align = False

else:
    raise ValueError('model not supported')



if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    #sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_set = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, vision="video", align=args.align, modality=args.modality, raw=args.raw_data)
    if args.use_dist:
        sampler = DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)

else:
    print('balanced sampler is not used')
    train_set = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, vision="video", align=args.align, modality=args.modality, raw=args.raw_data)
    if args.use_dist:
        sampler = DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_set = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, vision="video", align=args.align, modality=args.modality, raw=args.raw_data)
if args.use_dist:
    val_sampler = DistributedSampler(val_set)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=10,sampler=val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=10, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf, vision="video"),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



if args.pretrain_path == 'None':
    warnings.warn("Note you are finetuning a model without any finetuning.")

# finetune based on a CAV-MAE pretrained model, which is the default setting unless for ablation study
if args.pretrain_path != 'None':
    # TODO: change this to a wget link
    mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in mdl_weight.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # strip 'module.'
        else:
            new_state_dict[k] = v
    miss, unexpected = audio_model.load_state_dict(new_state_dict, strict=False)
    print('now load cav-mae pretrained weights from ', args.pretrain_path)
    print(miss, unexpected)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass

with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args, local_rank)

# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir, start_epoch, end_epoch):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location='cpu')
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location='cpu')
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA

# evaluate with multiple frames
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)
if args.wa == True:
    sdA = wa_model(args.exp_dir, start_epoch=args.wa_start, end_epoch=args.wa_end)
    torch.save(sdA, args.exp_dir + "/models/audio_model_wa.pth")
else:
    # if no wa, use the best checkpint
    sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')
msg = audio_model.load_state_dict(sdA, strict=True)
print(msg)
audio_model.eval()

# skil multi-frame evaluation, for audio-only model
if args.skip_frame_agg == True:
    val_audio_conf['frame_use'] = 5
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
    if args.metrics == 'mAP':
        cur_res = np.mean([stat['AP'] for stat in stats])
        print('mAP is {:.4f}'.format(cur_res))
    elif args.metrics == 'acc':
        cur_res = stats[0]['acc']
        print('acc is {:.4f}'.format(cur_res))
else:
    res = []
    multiframe_pred = []
    total_frames = 10 # change if your total frame is different
    for frame in range(total_frames):
        val_audio_conf['frame_use'] = frame
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
        print(audio_output.shape)
        if args.metrics == 'acc':
            audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
        elif args.metrics == 'mAP':
            audio_output = torch.nn.functional.sigmoid(audio_output.float())

        audio_output, target = audio_output.numpy(), target.numpy()
        multiframe_pred.append(audio_output)
        if args.metrics == 'mAP':
            cur_res = np.mean([stat['AP'] for stat in stats])
            print('mAP of frame {:d} is {:.4f}'.format(frame, cur_res))
        elif args.metrics == 'acc':
            cur_res = stats[0]['acc']
            print('acc of frame {:d} is {:.4f}'.format(frame, cur_res))
        res.append(cur_res)

    # ensemble over frames
    multiframe_pred = np.mean(multiframe_pred, axis=0)
    if args.metrics == 'acc':
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(multiframe_pred, 1))
        print('multi-frame acc is {:f}'.format(acc))
        res.append(acc)
    elif args.metrics == 'mAP':
        AP = []
        for k in range(args.n_class):
            # Average precision
            avg_precision = metrics.average_precision_score(target[:, k], multiframe_pred[:, k], average=None)
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print('multi-frame mAP is {:.4f}'.format(mAP))
        res.append(mAP)
    np.savetxt(args.exp_dir + '/mul_frame_res.csv', res, delimiter=',')