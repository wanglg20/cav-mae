import os
import torch
# from models import CAVMAE_Sync
# from models.cav_mae import CAVMAE_Sync_k700_FT
from traintest_ft import calculate_mAP

pred = torch.randn(10, 527)
target = torch.randn(10, 527)
mAP = calculate_mAP(pred, target)
print(mAP)