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
from models.videomamba_pretrain import PatchEmbed as PatchEmbedVideo, Linear_Decoder, get_sinusoid_encoding_table, _init_weights, create_block
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
            ssm_cfg=None, 
            norm_epsilon=1e-5,
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
            device=None,
            dtype=None,
            # Cross-Modality parameters
            use_global_pooling = False,
            
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
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )


        # General model parameters
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_frames = num_frames
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # cls token and position embedding
        self.patch_embed_a.num_patches = int(audio_length * 128 / 256) #(audio_len / 16) * (audio_channels / 16)
        self.num_patches_v = int(img_size * img_size / (patch_size * patch_size)) + 1 # H*W / (P*P), 196 = 14*14 by default; +1 represents the cls token 
        self.num_patches_a = self.patch_embed_a.num_patches + 1
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_global_pooling = use_global_pooling
        if self.use_global_pooling:               # use global pooling head to get global token
            self.global_pooling_head_v = GlobalPoolingHead(embed_dim)
            self.global_pooling_head_a = GlobalPoolingHead(embed_dim)
        else:                                   # use learnable global token
            self.global_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.global_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_patches_v += 1
            self.num_patches_a += 1
            trunc_normal_(self.cls_token_a, std=.02)
            trunc_normal_(self.cls_token_v, std=.02)

        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_v, embed_dim))
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches_a, embed_dim))
        self.temporal_pos_embed_v = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.temporal_pos_embed_a = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        pos_embed_v = get_sinusoid_encoding_table(
            self.num_patches_v, embed_dim, cls_token=True
        )
        pos_embed_a = get_sinusoid_encoding_table(
            self.num_patches_a, embed_dim, cls_token=True
        )
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float())
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float())
        self.temporal_pos_embed_v.data.copy_(torch.zeros(1, num_frames // kernel_size, embed_dim).float())
        self.temporal_pos_embed_a.data.copy_(torch.zeros(1, num_frames // kernel_size, embed_dim).float())

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
    
    def forward_features(self, x_v: Tensor, x_a: Tensor, mask=None) -> dict:
        """
        Forward pass for the CrossMamba model.
        Args:
            x_v (Tensor): Video input of shape [B, C, T, H, W].
            x_a (Tensor): Audio input of shape [B, C, T].
        Returns:
            dict: A dictionary containing the outputs for video and audio modalities.
        """
        B, C, T, H, W = x_v.shape

        # Patch embedding
        x_v = self.patch_embed_v(x_v)       # B, C_o, T_o, H_o, W_o, 
        B, C, T, H, W = x_v.shape
        x_v = x_v.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x_a = self.patch_embed_a(x_a)       # B, N, d
        B, N_a, d_a = x_a.shape
        x_a = x_a.reshape(B*T, -1, d_a)  # B*T, N_a // T, d_a

        # Add cls token
        cls_tokens_v = self.cls_token_v.expand(B*T, -1, -1)
        cls_tokens_a = self.cls_token_a.expand(B*T, -1, -1)
        x_v = torch.cat((cls_tokens_v, x_v), dim=1)  # B*T, N+1, d
        x_a = torch.cat((cls_tokens_a, x_a), dim=1)  # B*T, (N_a // T) +1, d_a

        # Add Global token
        if not self.use_global_pooling:
            global_token_v = self.global_token_v.expand(B*T, -1, -1)
            global_token_a = self.global_token_a.expand(B*T, -1, -1)
            x_v = torch.cat((x_v, global_token_v), dim=1)   # add global token at tail
            x_a = torch.cat((x_a, global_token_a), dim=1)
    
        # Add position embedding
        pos_embed_v = self.pos_embed_v.expand(B*T, -1, -1)
        pos_embed_a = self.pos_embed_a.expand(B*T, -1, -1)
        x_v = x_v + pos_embed_v
        x_a = x_a + pos_embed_a
        # Add temporal position embedding
        x_v = x_v.reshape(B, T, -1, x_v.shape[-1])
        B, T, N_v, D = x_v.shape
        temporal_pos_embed_v = self.temporal_pos_embed_v.unsqueeze(-2).expand(B, -1, N_v, -1)
        x_v = x_v + temporal_pos_embed_v
        x_a = x_a.reshape(B, T, -1, x_a.shape[-1])
        B, T, N_a, D = x_a.shape
        temporal_pos_embed_a = self.temporal_pos_embed_a.unsqueeze(-2).expand(B, -1, N_a, -1)
        x_a = x_a + temporal_pos_embed_a

        # Mask SSM pipeline
        x = torch.cat((x_v, x_a), dim=2)  # B, T, N+N_a, d
        x = x.reshape(B, -1, D)  # B, T*(N+N_a), d
        
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        x_clip_vis = []

        # mamba impl
        residual = None
        hidden_states = x_vis
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
            if (idx - 1) in self.return_index:
                x_clip_vis.append(self.norm(residual.to(dtype=self.norm.weight.dtype))) # share norm for mask

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if (self.depth - 1) in self.return_index:
            x_clip_vis.append(residual)
        x_clip_vis = torch.stack(x_clip_vis)

        return x_clip_vis
    

    def forward(self, x_v: Tensor, x_a: Tensor, mask=None) -> dict:
        x_clip_vis = self.forward_features(x_v, x_a, mask)     # Student features
        
        # align CLIP and ClAP
        K, B, _, C_CLIP = x_clip_vis.shape              # K represent the number of student layers to align with CLIP
        expand_clip_pos_embed = self.clip_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
        one = torch.ones((B, 1), device=mask.device, dtype=mask.dtype) # cls token is always visible
        mask = torch.cat((one, mask), dim=1) # [B, N+1, C_d_clip]
        clip_pos_emd_vis = expand_clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
        x_clip_full = x_clip_vis + clip_pos_emd_vis # [K, B, N, C_d_clip]

        x_clip = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip.append(clip_decoder(x_clip_full[idx]))
        x_clip = torch.stack(x_clip) # align and normalize

        return x_clip
