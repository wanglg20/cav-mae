# -*- coding: utf-8 -*-
# @Time    : 08/06/25 11:03 AM
# @Author  : Linge Wang

# inpaint video with video-pretrained model
# modified from 
# Author: Yuan Gong

import os.path
import torch
import models
import numpy as np
from matplotlib import pyplot as plt
import dataloader as dataloader

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def concat_video(raw_video):
    """
    concat a video to a single image with frames laid horizontally and separated by black lines.

    :param raw_video: Tensor of shape (T, 3, H, W)
    """
    if isinstance(raw_video, torch.Tensor):
        raw_video = raw_video.detach().cpu().numpy()

    T, C, H, W = raw_video.shape

    # Convert to (T, H, W, C) and scale to [0, 1] if necessary
    video_np = np.transpose(raw_video, (0, 2, 3, 1))  # (T, H, W, C)
    # if video_np.max() > 1.0:
    #     video_np = video_np / 255.0
    video_np = np.clip((video_np * imagenet_std + imagenet_mean) * 255, 0, 255)
    video_np = video_np.astype(np.uint8)  # Ensure it's in uint8 format
    # Create black separator lines
    line_width = 2  # in pixels
    black_line = np.zeros((H, line_width, 3), dtype=video_np.dtype)

    # Concatenate frames with black lines
    frames_with_lines = []
    for i, frame in enumerate(video_np):
        frames_with_lines.append(frame)
        if i != T - 1:  # no black line after the last frame
            frames_with_lines.append(black_line)

    # Concatenate all along width
    combined = np.concatenate(frames_with_lines, axis=1)  # horizontal concat
    return combined

def show_image(image, title=''):
    # image is [H, W, 3]
    if image.shape[2] == 3:
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    else:
        plt.imshow(image, origin='lower')
    plt.title(title, fontsize=12)
    plt.axis('off')
    return

def run_one_video(audio, video, model, mask_ratio_a=0.75, mask_ratio_v=0.75, mask_mode='unstructured'):
    v = video
    B, T, C, H, W = v.shape
    pred_a, pred_v, mask_a, mask, loss_a, loss_v = model.module.forward_inpaint(audio, v.float(), mask_ratio_a=mask_ratio_a, mask_ratio_v=mask_ratio_v)

    pred_a = model.module.unpatchify(pred_a, 1, 8, 64, 16)  # B, C, H, W
    pred_a = torch.einsum('nchw->nhwc', pred_a).detach().cpu()  # B, H, W, C
    mask_a = mask_a.detach()
    mask_a = mask_a.unsqueeze(-1).repeat(1, 1, model.module.patch_embed_a.patch_size[0] ** 2 * 1)  # (N, H*W, p*p*3)
    # mask_a = model.module.unpatchify(mask_a, 1, 8, 64, 16)  # 1 is removing, 0 is keeping
    mask_a = model.module.unpatchify(mask_a, 1, 8, 64, 16)  # 1 is removing, 0 is keeping
    mask_a = torch.einsum('nchw->nhwc', mask_a).detach().cpu()
    audio = torch.einsum('nchw->nwhc', audio.unsqueeze(0))  # B, H, W, C
    audio_masked = audio * (1 - mask_a)
    audio_paste = audio * (1 - mask_a) + pred_a * mask_a

    print("Loss A: {:.4f}, Loss V: {:.4f}".format(loss_a.item(), loss_v.item()))
    fig_a = plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    show_image(audio[0], "Original Audio")
    plt.subplot(1, 3, 2)
    show_image(audio_masked[0], "Masked Audio")
    plt.subplot(1, 3, 3)
    show_image(audio_paste[0], "Reconstructed Audio")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'audio_recons.png'))
    plt.title('Audio Reconstruction', fontsize=16)

    plt.close(fig_a)

    B, N, D = pred_v.shape # B, 1960, D, 1960 = 196 * 10 = num_patches_per_frame * num_frames
    pred_v = pred_v.reshape(B*T, -1, D)
    pred_v = model.module.unpatchify(pred_v, 3, 14, 14, 16) # B*T, C, H, W
    pred_v = pred_v.reshape(B, T, C, H, W)

    mask = mask.detach()            #B, 1960
    mask = mask.reshape(B, T, -1)  # B, T, 196
    mask = mask.reshape(B*T, -1)  # B*T, 196
    mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_embed_v.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3) B, 196, 768
    mask = model.module.unpatchify(mask, 3, 14, 14, 16)  # B*T, C, H, W
    mask = mask.reshape(B, T, C, H, W)  # B, T, C, H, W



    video_masked = v * (1 - mask)
    video_paste = v * (1 - mask) + pred_v * mask

    plot_ori_v = concat_video(v[0].detach().cpu().numpy())
    fig_test = plt.figure(figsize=(12, 12))
    plt.imshow(plot_ori_v)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'video_ori.png'))
    plot_masked_v = concat_video(video_masked[0].detach().cpu().numpy())
    plot_recons_v = concat_video(video_paste[0].detach().cpu().numpy())

    fig_v = plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.imshow(plot_ori_v)
    plt.title('Original Video', fontsize=16)
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(plot_masked_v)
    plt.title('Masked Video', fontsize=16)
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(plot_recons_v)
    plt.title('Reconstructed Video', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'video_recons.png'))
    plt.title('Video Reconstruction', fontsize=16)
    plt.close(fig_v)




device = "cpu"
mask_ratio_a, mask_ratio_v = 0.75, 0.75
# or 'time' or 'freq' or 'tf'
mask_mode = 'unstructured'
# the model has to be trained without pixel normalization for inpaint purpose
model_path = '/data/wanglinge/project/cav-mae/src/exp/pretrain-videomae-audioset-cav-mae-lr5e-5-bs6-normFalse-c0.01-p1.0-tpFalse-mr-unstructured-0.75/models/audio_model.20.pth'
save_dir = 'vis/videomae_recons'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

videomae = models.CAVMAE(video_input=True)
videomae = torch.nn.DataParallel(videomae)
msg = videomae.load_state_dict(torch.load(model_path, map_location=device), strict=False)
videomae.to(device)
videomae.eval()

audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'audioset', 'mode':'train', 'mean':-5.081, 'std':4.4849,
              'noise':False, 'label_smooth': 0, 'im_res': 224}

dataset = dataloader.AudiosetDataset('/data/wanglinge/project/cav-mae/src/data/info/k700_test.json', audio_conf, 
                               label_csv='/data/wanglinge/project/cav-mae/src/data/info/k700_class.csv',  modality='both', raw='k700', vision='video')

val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
A_loss_a, A_loss_v = [], []


for i, (a_input, v_input, _) in enumerate(val_loader):
    a_input = a_input.to(device)
    v_input = v_input.to(device)
    # v_input = v_input[0].permute(1, 2, 0)
    # a_input = a_input.permute(1, 2, 0)
    # pred_a, pred_v, mask_a, mask, loss_a, loss_v = videomae.module.forward_inpaint(a_input, v_input, mask_ratio_a=mask_ratio_a, mask_ratio_v=mask_ratio_v)
    # A_loss_a.append(loss_a.item())
    # A_loss_v.append(loss_v.item())
    run_one_video(a_input, v_input, videomae, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
    break


print("Average Loss A: {:.4f}, Loss V: {:.4f}".format(np.mean(A_loss_a), np.mean(A_loss_v)))