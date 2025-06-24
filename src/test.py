import os
import torch
from collections import OrderedDict

import torch
from dataloader import rand_mask_generate, mask_expand2d
import numpy as np
from engine_mamba_training import attn_mask_generator, resume_training

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
a = a.flatten(0)
print(a.shape)
print(a)
print(a[::4])