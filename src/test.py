import os
import torch
from collections import OrderedDict

import torch

x = torch.arange(2 * 32).view(2, 32)  # 示例数据

# 先在行方向重复（B 方向扩展为 4）
x_expanded = x.repeat_interleave(2, dim=0)

# 再在列方向重复（列扩展为 64）
x_expanded = x_expanded.repeat_interleave(2, dim=1)

print(x_expanded.shape)  # torch.Size([4, 64])
print(x_expanded)
