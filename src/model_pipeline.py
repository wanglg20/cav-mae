import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
from models.mamba_pretrain import CrossMamba
from models.videomamba_pretrain import VisionMamba
from models.teacher import clip_b16
from dataloader import rand_mask_generate, mask_expand2d
from transformers.models.clap.modeling_clap import ClapAudioModelOutput, ClapAudioPatchEmbed, ClapAudioStage, ClapAudioPatchMerging
from transformers.models.clap.modeling_clap import ClapAudioModel
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor

def test_cross_mamba():
    model = CrossMamba(
        num_frames=16,
        audio_length=1024,
        )
    print("MambaPretrain model created successfully.")
    print(model.patch_embed_v.num_patches, model.patch_embed_a.num_patches)
    
    import torch
    B = 2
    v = torch.randn(B, 3, 16, 224, 224)  # Video input
    a = torch.randn(B, 64, 1024)
    zeros = torch.zeros(16, 1).bool()
    mask_v = rand_mask_generate(16, 196, 0.75)
    mask_a = rand_mask_generate(16, 64 // 16, 0.75)
    mask_a_ori = mask_a.reshape(16, 2, 2)
    mask_a = mask_expand2d(mask_a_ori, expand_ratio=2)  # Frame, Freq, Time, 
    mask_a = mask_a.reshape(16, -1)
    mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=1)
    mask = mask.reshape(1, -1)
    mask = mask.repeat(B, 1)  # Repeat for batch size
    mask_v = mask_v.reshape(1, -1).repeat(B, 1)
    mask_a_ori = mask_a_ori.reshape(1, -1).repeat(B, 1)
      # Add ones at the beginning and end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    v = v.to(device)
    a = a.to(device)
    mask = mask.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {} M'.format(n_parameters / 1e6))
    clip_vis = model.forward_features(v, a, mask)
    x_clip, x_clap, global_v, global_a = model(v, a, mask)

    print("CLIP output shape:", x_clip.shape)           #1, 1, 784, 512
    print("CLAP output shape:", x_clap.shape)           #1, 1, 64, 512
    print("Global Video shape:", global_v.shape)         #1, 16
    print("Global Audio shape:", global_a.shape)         #1, 16,
    pred_clap = x_clap[:, :, ::4, :]
    print("Pred CLAP shape:", pred_clap.shape)         #1, 1, 16, 512
    teacher_clip = torch.randn(B, 196*16, 768).to(device)  # Simulated teacher output
    teacher_clap = torch.randn(B, 64, 768).to(device)
    
    mask = mask.reshape(B, 16, -1)
    mask_v = mask_v.reshape(B, -1)
    mask_a = mask_a_ori.reshape(B, -1)
    target_clip = teacher_clip[~mask_v].reshape(B, -1, 768)
    target_clap = teacher_clap[~mask_a].reshape(B, -1, 768)
    print("Target CLIP shape:", target_clip.shape)     # 1, 196*16, 768
    print("Target CLAP shape:", target_clap.shape)     # 1, 16, 768

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
    # weight = '/data/wanglinge/project/cav-mae/src/weight/teacher/vit_b16.pth'
    # teacher_model.load_state_dict(torch.load(weight, map_location='cpu'), strict=True)
    mask = torch.cat([
        torch.ones(1, 10 * int(14 * 14 * 0.75)),
        torch.zeros(1, 10 * int(14 * 14 * 0.25)),
    ], dim=-1).bool()
    mask = mask.repeat(2, 1)
    x = torch.randn(2, 3, 16, 224, 224)  
    out = teacher_model(x)
    print(out[0].shape)  # K, B, 1961, 768
    print(out[1].shape)  # Attention weights shape  B*T, 49


def test_clap_teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
    clap_encoder = clap_model.audio_model
    weight_path = '/data/wanglinge/project/cav-mae/src/weight/teacher/clap.pth'
    clap_encoder.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    teacher_model = clap_encoder
    audio = torch.randn(1, 1, 1024, 64).to(device)  # 假设的输入音频特征, B
    teacher_model = teacher_model.to(device)
    out = teacher_model(audio, is_longer=torch.tensor([1]).bool().to(device), output_attentions=torch.tensor([True]).bool().to(device), return_dict=True)
    clap_feat = out.last_hidden_state  # 1, 768, 2, 32
    clap_attn = out.attentions[-1]  # 1, 32, 64, 64
    print(clap_feat.shape)  # 1, 768, 2, 32
# def test_clap_teacher():
    

if __name__ == "__main__":
    test_clip_teacher()
    test_cross_mamba()
    test_clap_teacher()