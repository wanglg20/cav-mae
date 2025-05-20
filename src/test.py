import os
import torch
from models import CAVMAE_Sync
from models.cav_mae import CAVMAE_Sync_k700_FT

exp_dir = "/data/wanglinge/project/cav-mae/src/exp/trainmae-audioset-cav-mae-balNone-lr5e-5-epoch25-bs60-normFalse-c0.01-p1.0-tpFalse-mr-unstructured-0.75"
ckpt_list = os.listdir(os.path.join(exp_dir, "models"))
epochs = [int(f.split('.')[-2]) for f in ckpt_list if f.endswith('.pth') and f.startswith('audio_model')]
print(epochs)

if __name__ =='__main__':
    model = CAVMAE_Sync_k700_FT(label_dim=700, img_size=224, audio_length=400, patch_size=16, in_chans=3,)
    audio = torch.randn(10, 128, 400)
    vision = torch.randn(10, 3, 224, 224)
    pred = model(audio, vision)
    print(pred.shape)
    
    # loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = model(audio, vision)
    # print(loss.shape)
    