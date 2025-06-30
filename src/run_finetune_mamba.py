import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os
import ast
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

# dist and log:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import socket

# borrowed model and training engine
from models.mamba_pretrain import CrossMamba, CrossMambaFT, load_from_pretrained
from engine_mamba_training import finetune_mamba
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


# finetune cav-mae model
def arg_parser():
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

    parser.add_argument("--visible_gpus", type=str, default='0,1,2,3,4,5,6,7')
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
    parser.add_argument("--raw_data", type=str, default="k700", help="raw data of daataset")
    parser.add_argument("--train_frame_root", type=str, default='/data/wanglinge/project/cav-mae/src/data/k700/train_16f', help="the root directory for training video frames")
    parser.add_argument("--val_frame_root", type=str, default='/data/wanglinge/project/cav-mae/src/data/k700/val_16f', help="the root directory for validation video frames")
    args = parser.parse_args()
    return args
def main():
    args = arg_parser()
    local_rank = setup_distributed(args.visible_gpus)
    # Distributed training setup And Wandb Setup
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
                # wandb.init(project=args.wandb_project_name,
                #        entity='wanglg-institude-of-automation-cas',
                #        notes=socket.gethostname(),
                #        name='cav_1',
                #        job_type="training",
                #        reinit=True)
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

    im_res = 224
    audio_conf = {'num_mel_bins': 64, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
                  'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                  'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res}
    val_audio_conf = {'num_mel_bins': 64, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                      'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
    model = CrossMambaFT(
        num_classes=700, fc_drop_rate=0.1
    )
    missing_keys, unexpected_keys = load_from_pretrained(model, args.pretrain_path, strict=False)
    print(("newly initialized keys: ", missing_keys))
    print(("unexpected keys: ", unexpected_keys))

    train_set = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, 
        audio_conf=audio_conf, modality='both', vision='video', raw='k700', num_frames=16,
        use_mask=True, video_frame_dir=args.train_frame_root)
    sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_set = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, num_frames=16,
        audio_conf=val_audio_conf, modality='both', vision='video', raw='k700', 
        use_mask=True, video_frame_dir=args.val_frame_root)
    val_sampler = DistributedSampler(val_set)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=10, sampler = val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("\nCreating experiment directory: %s" % args.exp_dir)
    try:
        os.makedirs("%s/models" % args.exp_dir)
    except:
        pass

    print('Now starting training for {:d} epochs.'.format(args.n_epochs))
    finetune_mamba(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        args=args,
        local_rank=local_rank,
    )

if __name__ == "__main__":
    main()
