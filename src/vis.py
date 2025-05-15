# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

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
import numpy as np
from traintest_cavmae import train

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import socket



# finetune cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700_val.json', help="training data json")
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
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
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
parser.add_argument("--wandb_run_name", type=str, default='cav_baseline')
parser.add_argument("--use_wandb", action="store_true",
                        help="use wandb or not")
args = parser.parse_args()

if __name__ == '__main__':
    audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'audioset', 'mode':'train', 'mean':-5.081, 'std':4.4849,
              'noise':True, 'label_smooth': 0, 'im_res': 224}
    dataset = dataloader.AudiosetDataset('/data/wanglinge/project/cav-mae/src/data/info/k700_val.json', audio_conf, label_csv='data/info/k700_class.csv')
    print('dataset length is {:d}'.format(len(dataset)))
    
    train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = models.CAVMAE(audio_length=args.target_length, norm_pix_loss=args.norm_pix_loss, modality_specific_depth=11, tr_pos=args.tr_pos)
    weight = '/data/wanglinge/project/cav-mae/src/exp/testmae-audioset-cav-mae-balNone-lr5e-5-epoch25-bs60-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75/models/audio_model.25.pth'
    model_weight = torch.load(weight, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in model_weight.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # strip 'module.'
        else:
            new_state_dict[k] = v
    miss, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Miss:", miss)
    print("unexpect:", unexpected)

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    def save_image_tensor(image_tensor, save_path):
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # 归一化到[0,1]
        image_uint8 = (image_np * 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(save_path)
        print(f"Image saved to {save_path}")
    def save_spectrogram_tensor(spec_tensor, save_path):
        spec_np = spec_tensor.squeeze(0).cpu().detach().numpy()
        spec_db = 10 * np.log10(np.maximum(spec_np, 1e-10))  # 转换为 dB 单位

        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db, origin='lower', aspect='auto', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Spectrogram saved to {save_path}")

    for i, (a_input, v_input, _) in enumerate(train_loader):
        if i<3:
            continue
        save_image_tensor(v_input, "ori_v.png")
        save_spectrogram_tensor(a_input[0], "ori_a.png")
        pred_a, pred_v, mask_a, mask_v, loss_pixel_a, loss_pixel_v = model.forward_inpaint(audio=a_input, imgs=v_input)
        print("audio:")
        print(mask_a.shape)
        print(pred_a.shape)
        print("imgs:")
        print(mask_v.shape)
        print(pred_v.shape)
        # patch: 1, 8, 64, 16
        # 3, 14, 14, 16
        img = model.patchify(v_input, 3, 14, 14, 16)
        mean = img.mean(dim=-1, keepdim=True)
        var = img.var(dim=-1, keepdim=True)
        mask_v = mask_v.unsqueeze(-1)
        pred_v = pred_v * var + mean
        pred_v = img * (1 - mask_v) + pred_v * mask_v
        pred_v = model.unpatchify(pred_v, c=3, h=14, w=14)

        audio = model.patchify(a_input, 1, 8, 64, 16)
        mean = audio.mean(dim=-1, keepdim=True)
        var = audio.var(dim=-1, keepdim=True)
        mask_a = mask_a.unsqueeze(-1)
        pred_a = pred_a * var + mean
        pred_a = audio * (1 - mask_a) + pred_a * mask_a
        pred_a = model.unpatchify(pred_a, 1, 8, 64, 16)
        # print(pred_v.shape)
        # print(pred_a.shape)

        save_image_tensor(pred_v, "recons_v.png")
        save_spectrogram_tensor(pred_a[0], "recos_a.png")
        print(loss_pixel_a)
        print(loss_pixel_v)
        break

