import clip.clip as clip
import os
import torch
from collections import OrderedDict

path = 'weight/teacher'
model, _ = clip.load("ViT-B/16", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(path, 'vit_b16.pth'))