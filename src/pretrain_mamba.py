import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5" 

import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
from models.videomamba_pretrain import VisionMamba
from models.mamba_pretrain import CrossMamba
from models.videomamba_pretrain import VisionMamba
from transformers import ClapModel, ClapProcessor
import numpy as np
from engine_mamba_training import train_mamba

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import socket
from models.teacher import clip_b16


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
    # available gpus:
    print("available gpus: ", visible_devices)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    setup_for_distributed(local_rank == 0)
    return local_rank
# pretrain cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_train.json', help="training data json")
parser.add_argument("--data-val", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_test.json', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_class.csv', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=700, help="number of classes")
parser.add_argument("--model", type=str, default='cav-mae', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt"])
parser.add_argument("--dataset_mean", type=float, default=-5.081,help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=4.4849,help="the dataset audio spec std, used for input normalization")
parser.add_argument("--target_length", type=int, default=1024, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=4, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments, only for preliminary experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
parser.add_argument("--weight_file", type=str, default='/data/wanglinge/project/cav-mae/src/weight/init/ori_mae_11.pth', help="path to weight file")
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=True)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=False)
parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
parser.add_argument("--mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])
parser.add_argument("--visible_gpus", type=str, default='0,1,2,3,4,5')
parser.add_argument("--wandb_project_name", type=str, default='cav')
parser.add_argument("--wandb_run_name", type=str, default='pretrain_mamba')
parser.add_argument("--use_wandb", action="store_true",
                        help="use wandb or not")
parser.add_argument("--wandb_id", type=str, default=None,
                        help="wandb id if resuming from a previous run")                        
parser.add_argument("--resume", action="store_true",
                        help="resume from a previous run")
parser.add_argument("--contrastive_loss_weight", type=float, default=0.5, help="weight for contrastive loss")
parser.add_argument("--mask_ratio", type=float, default=0.75, help="masking ratio for contrastive loss")

args = parser.parse_args()
im_res = 224

local_rank = setup_distributed(args.visible_gpus)
audio_conf = {'num_mel_bins': 64, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))


print(args)
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
            os.environ["WANDB_DIR"] = "./wandb_offline"
            wandb.init( project=args.wandb_project_name,
               entity='wanglg-institude-of-automation-cas',
               notes=socket.gethostname(),
               name='cav_1',
               job_type="training",
               reinit=True,
               mode="offline" )
        if args.wandb_run_name != None:
            wandb.run.name = args.wandb_run_name
        wandb.config.update(args)

model = CrossMamba(
        num_frames=16,
        audio_length=1024,
        )
teacher_v = clip_b16(
      pretrained=True,
      clip_norm_type='l2',
      input_resolution=224,
      return_attn=True,
      clip_return_layer=1,
      clip_return_interval=1,
      clip_return_cls=True
    )
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused")
clap_encoder = clap_model.audio_model
weight_path = '/data/wanglinge/project/cav-mae/src/weight/teacher/clap.pth'
clap_encoder.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
teacher_a = clap_encoder


n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params: {} M'.format(n_parameters / 1e6))

# Dataset and Dataloader
train_set = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, 
    audio_conf=audio_conf, modality='both', vision='video', raw='k700', num_frames=16,
    use_mask=True, video_frame_dir='/data/wanglinge/project/cav-mae/src/data/k700/train_16f')
sampler = DistributedSampler(train_set)
train_loader = torch.utils.data.DataLoader(
train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_set = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, num_frames=16,
    audio_conf=val_audio_conf, modality='both', vision='video', raw='k700', 
    use_mask=True, video_frame_dir='/data/wanglinge/project/cav-mae/src/data/k700/val_16f')
val_sampler = DistributedSampler(val_set)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=10, sampler = val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train_mamba(model, teacher_v, teacher_a, train_loader, val_loader, args, local_rank )