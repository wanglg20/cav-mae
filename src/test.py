import os
import torch
from collections import OrderedDict

import torch
from dataloader import rand_mask_generate, mask_expand2d
import numpy as np

def rand_mask_generate(num_frames, num_patches, mask_ratio):
    num_mask_per_frame = int(num_patches * mask_ratio)

    rand_vals = torch.rand((num_frames, num_patches), device='cpu')
    _, indices = torch.topk(rand_vals, k=num_mask_per_frame, dim=1, largest=False)
    mask = torch.zeros((num_frames, num_patches), dtype=torch.bool, device='cpu')
    mask.scatter_(1, indices, True)
    # mask = mask.flatten()
    return mask


zeros = torch.zeros(16, 1).bool()
mask_v = rand_mask_generate(16, 196, 0.75)
mask_a = rand_mask_generate(16, 64 // 16, 0.75)
mask_a = mask_a.reshape(16, 2, 2)
mask_a = mask_expand2d(mask_a, expand_ratio=2)  # Frame, Freq, Time, 
mask_a = mask_a.reshape(16, -1)
mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=1)
print(mask.shape)  # Should be (16, 784)
print(mask[0])