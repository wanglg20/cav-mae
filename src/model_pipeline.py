import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
from models.mamba_pretrain import CrossMamba
from models.videomamba_pretrain import VisionMamba
from models.teacher import clip_b16

def test_cross_mamba():
    model = CrossMamba(
        num_frames=16,
        audio_length=1024,
        )
    print("MambaPretrain model created successfully.")
    print(model.patch_embed_v.num_patches, model.patch_embed_a.num_patches)
    
    import torch
    # v = torch.randn(1, 3, 10, 224, 224)  # Video input
    # a = torch.randn(1, 64, 960)
    # zeros = torch.zeros(1, 1)
    # mask = torch.cat([
    #     zeros,
    #     torch.zeros(1,  int(14 * 14 * 0.25)),
    #     torch.ones(1,  int(14 * 14 * 0.75)),
    #     zeros,
    #     zeros,
    #     torch.ones(1, int(6 * 4 * 0.75)),      # # 6 = 960 / 10 / 16(10 = num_frames, 16 = patch_size); 8 = 128 / 16(4 = 64 / 16), 128 = num_mel_bins
    #     torch.zeros(1, int(6 * 4 * 0.25)),
    #     zeros,
    # ], dim=-1).to(torch.bool)
    # mask = mask.squeeze(1).repeat(1, 10, 1)
    # mask = mask.reshape(1, -1)
    v = torch.randn(1, 3, 16, 224, 224)  # Video input
    a = torch.randn(1, 64, 1024)
    zeros = torch.zeros(1, 1)
    mask = torch.cat([
        zeros,
        torch.zeros(1,  int(14 * 14 * 0.25)),
        torch.ones(1,  int(14 * 14 * 0.75)),
        zeros,
        zeros,
        torch.ones(1, int(4 * 4 * 0.75)),      # # 6 = 960 / 10 / 16(10 = num_frames, 16 = patch_size); 8 = 128 / 16(4 = 64 / 16), 128 = num_mel_bins
        torch.zeros(1, int(4 * 4 * 0.25)),
        zeros,
    ], dim=-1).to(torch.bool)
    mask = mask.squeeze(1).repeat(1, 16, 1)
    mask = mask.reshape(1, -1)
      # Add ones at the beginning and end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    v = v.to(device)
    a = a.to(device)
    mask = mask.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {} M'.format(n_parameters / 1e6))
    clip_vis = model.forward_features(v, a, mask)
    x_clip, x_clap = model(v, a, mask)
    print("CLIP output shape:", x_clip.shape)           #1, 1, 784, 512
    print("CLAP output shape:", x_clap.shape)
    teacher_clip = torch.randn(1, 490, 768).to(device)  # Simulated teacher output
    teacher_clap = torch.randn(1, 64, 768).to(device)


def test_clip_teacher():
    teacher_model = clip_b16(
      pretrained=True,
      clip_norm_type='l2',
      input_resolution=224,
      return_attn=True,
      clip_return_layer=1,
      clip_return_interval=1,
      clip_return_cls=True
    )
    mask = torch.cat([
        torch.ones(1, 10 * int(14 * 14 * 0.75)),
        torch.zeros(1, 10 * int(14 * 14 * 0.25)),
    ], dim=-1).bool()
    x = torch.randn(1, 3, 10, 224, 224)  
    out = teacher_model(x, mask)
    print(out[0].shape)        


# def test_clap_teacher():
    

if __name__ == "__main__":
    #key_mapping(torch.load('weight/teacher/clip_vit_b16.pth', map_location='cpu'))
    # test_clip_teacher()
    test_cross_mamba()
