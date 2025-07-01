import torch
from collections import OrderedDict
import numpy as np
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
# print(clap_model.state_dict().keys())
new_state_dict = OrderedDict()
path = 'weight/teacher'
# 提取 'audio_model' 部分的所有权重
for k, v in clap_model.state_dict().items():
    if 'audio_model.' in k:
        # 去掉 'audio_model.' 前缀，并将其加入新的 OrderedDict
        new_state_dict[k[12:]] = v

# 保存提取的音频模型权重到文件
torch.save(new_state_dict, os.path.join(path, 'clap.pth'))

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