import os
import torch
from collections import OrderedDict

import torch
from dataloader import rand_mask_generate, mask_expand2d
import numpy as np
from engine_mamba_training import attn_mask_generator, resume_training

x = torch.randn(16, 16)
mask = attn_mask_generator(x, 0.75, 16)
print(mask.shape)  # Should print (16, 16)