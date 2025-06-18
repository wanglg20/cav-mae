# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
import numpy as np
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from models.videomamba_pretrain import Block
from models.videomamba_pretrain import PatchEmbed as PatchEmbedVideo, Linear_Decoder, get_sinusoid_encoding_table, _init_weights
from models.cav_mae import PatchEmbed as PatchEmbedImg

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class GlobalPoolingHead(nn.Module):
    """
    Global pooling head that computes a global representation of the input sequence.
    This head uses a learnable query vector to compute the global representation
    Args:
        dim (int): Dimension of the input features.
        heads (int): Number of attention heads to use for pooling.
    Returns:
        g (Tensor): Global representation of shape [B, D], where B is the batch
        size and D is the feature dimension.
    """
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        # 用线性投影自己实现 QKV，节省多余 bias / dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def forward(self, h: Tensor) -> Tensor:
        """
        h: [B, L, D]
        return g: [B, D]
        """
        B, L, D = h.shape
        q = self.q_proj(self.query).expand(B, -1, -1)          # [B, 1, D]
        k = self.k_proj(h)                                     # [B, L, D]
        v = self.v_proj(h)                                     # [B, L, D]

        # reshape to [B, heads, L, d_head]
        def split(x):  # [B, T, D] -> [B, heads, T, d_head]
            return x.view(B, -1, self.heads, D // self.heads).transpose(1, 2)

        q, k, v = map(split, (q, k, v))                        # q:[B,h,1,d] k,v:[B,h,L,d]
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B,h,1,L]
        attn = attn.softmax(dim=-1)
        g = (attn @ v).transpose(1, 2).reshape(B, 1, D)        # [B,1,D]
        return g.squeeze(1)                                    # [B,D]


class CrossMamba(nn.Module): 
    def __init__(
            self,
            # Patch embedding parameters
            img_size=224,
            audio_length=1024,
            patch_size=16,
            embed_dim=768,
            kernel_size=1,
            # Mamba parameters
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            initializer_cfg=None,
            # CLIP and clap decoder parameters# clip,
            clip_decoder_embed_dim=768,
            clap_output_dim=512,
            clip_output_dim=512,
            clip_norm_type='l2',
            clip_return_layer=1,
            clip_student_return_interval=1,
            # Model general parameters
            depth=12,
            drop_path_rate=0.,
            num_frames=8,
            # Cross-Modality parameters
            use_global_token = True,
            
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed_v = PatchEmbedVideo(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=3,
            kernel_size=kernel_size,
        )
        self.patch_embed_a = PatchEmbedImg(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=1,
        )

        # Mamba parameters
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.bimamba = bimamba

        # General model parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_frames = num_frames
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # cls token and position embedding
        self.patch_embed_a.num_patches = int(audio_length * 128 / 256) #(audio_len / 16) * (audio_channels / 16)
        self.num_patches_v = self.patch_embed_v.num_patches
        self.num_patches_a = self.patch_embed_a.num_patches
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.use_global_token = use_global_token
        if self.use_global_token:
            self.global_pooling_head_v = GlobalPoolingHead(embed_dim)
            self.global_pooling_head_a = GlobalPoolingHead(embed_dim)
            self.num_patches_v += 1
            self.num_patches_a += 1

        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_v + 1, embed_dim))
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches_a + 1, embed_dim))

        # CLIP decoder
        self.clip_decoder = nn.ModuleList([
            Linear_Decoder(
                output_dim=clip_output_dim, 
                embed_dim=clip_decoder_embed_dim, 
                norm_layer=nn.LayerNorm, 
                clip_norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])
        self.clap_decoder = nn.ModuleList([
            Linear_Decoder(
                output_dim=clap_output_dim, 
                embed_dim=clip_decoder_embed_dim, 
                norm_layer=nn.LayerNorm, 
                clip_norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])
        self.clip_pos_embed = get_sinusoid_encoding_table(
            (self.num_patches_v + 1) * num_frames // kernel_size + 1, 
            clip_decoder_embed_dim
        )
        self.clap_pos_embed = get_sinusoid_encoding_table(
            (self.num_patches_a + num_frames) // kernel_size + 1, 
            clip_decoder_embed_dim
        )


        # init
        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed_a, std=.02)
        trunc_normal_(self.pos_embed_v, std=.02)
        trunc_normal_(self.cls_token_a, std=.02)
        trunc_normal_(self.cls_token_v, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    
    def forward(self, x_v: Tensor, x_a: Tensor) -> dict:
        """
        Forward pass for the CrossMamba model.
        Args:
            x_v (Tensor): Video input of shape [B, C, T, H, W].
            x_a (Tensor): Audio input of shape [B, C, T].
        Returns:
            dict: A dictionary containing the outputs for video and audio modalities.
        """
        B = x_v.shape[0]
        
        # Patch embedding
        x_v = self.patch_embed_v(x_v)
        x_a = self.patch_embed_a(x_a)

        