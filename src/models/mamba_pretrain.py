# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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

def load_from_pretrained(model, pretrained_path: str, strict: bool = False):
        """
        Load model weights from a pre-trained CrossMamba checkpoint.
        
        Args:
            pretrained_path (str): Path to the pre-trained model checkpoint.
        """
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Remove Decoder-related keys
        keys_to_remove = [
            'clip_decoder', 
            'clap_decoder', 
            'clip_pos_embed', 
            'clap_pos_embed', 
            'return_index'
        ]
        for key in keys_to_remove:
            state_dict = {k: v for k, v in state_dict.items() if key not in k}
        # Load the state_dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        return missing_keys, unexpected_keys

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
            audio_length=1024,               # 960 mod 16*10 = 0, ensure audio length is divisible by 16*10
            num_bins = 64,
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
            clap_output_dim=768,
            clip_output_dim=512,
            clip_norm_type='l2',
            clip_return_layer=1,
            clip_student_return_interval=1,
            # Model general parameters
            depth=32,
            drop_path_rate=0.4,
            num_frames=16,
            device=None,
            dtype=None,
            # Cross-Modality parameters
            use_global_pooling = False,
            
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
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
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
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
                    #**factory_kwargs,
                )
                for i in range(depth)
            ]
        )


        # General model parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_frames = num_frames
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # cls token and position embedding
        self.patch_embed_a.num_patches = int(audio_length * num_bins / 256) #(audio_len / 16) * (audio_channels / 16)
        self.num_patches_v = int(img_size * img_size / (patch_size * patch_size)) # H*W / (P*P), 196 = 14*14 by default; 
        self.num_patches_a = self.patch_embed_a.num_patches // num_frames 
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_global_pooling = use_global_pooling
        if self.use_global_pooling:               # use global pooling head to get global token
            self.global_pooling_head_v = GlobalPoolingHead(embed_dim)
            self.global_pooling_head_a = GlobalPoolingHead(embed_dim)
        else:                                   # use learnable global token
            self.global_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.global_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # self.num_patches_v += 1
            # self.num_patches_a += 1
            trunc_normal_(self.cls_token_a, std=.02)
            trunc_normal_(self.cls_token_v, std=.02)

        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_v, embed_dim))
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches_a, embed_dim))
        self.temporal_pos_embed_v = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.temporal_pos_embed_a = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        pos_embed_v = get_sinusoid_encoding_table(self.num_patches_v, embed_dim)
        pos_embed_a = get_sinusoid_encoding_table(self.num_patches_a, embed_dim)
        self.pos_embed_v.data.copy_(pos_embed_v.float())
        self.pos_embed_a.data.copy_(pos_embed_a.float())
        self.temporal_pos_embed_v.data.copy_(torch.zeros(1, num_frames // kernel_size, embed_dim).float())
        self.temporal_pos_embed_a.data.copy_(torch.zeros(1, num_frames // kernel_size, embed_dim).float())

        # output head:
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

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
            (self.num_patches_v) * num_frames // kernel_size, 
            clip_decoder_embed_dim
        )
        self.clap_pos_embed = get_sinusoid_encoding_table(
            self.num_patches_a * num_frames // kernel_size, 
            clip_decoder_embed_dim
        )
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        

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
            x_a (Tensor): Audio input of shape [B, num_bins, T].
        Returns:
            dict: A dictionary containing the outputs for video and audio modalities.
        """
        B, C, T, H, W = x_v.shape

        # Patch embedding
        x_v = self.patch_embed_v(x_v)       # B, C_o, T_o, H_o, W_o, 
        B, C, T, H, W = x_v.shape
        x_v = x_v.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x_a = x_a.unsqueeze(1)              # B, 1, num_bins, T
        x_a = x_a.transpose(2, 3)           # B, 1, T, num_bins
        x_a = self.patch_embed_a(x_a)       # B, N_a, d_a
        B, N_a, d_a = x_a.shape
        x_a = x_a.reshape(B*T, -1, d_a)     # B*T, N_a // T, d_a
    
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
        x_v = x_v.reshape(B*T, N_v, D)       # B*T, N_v, d
        x_a = x_a.reshape(B*T, N_a, d_a)     # B

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

        x_v, x_a = x_v.reshape(B, T, -1, D), x_a.reshape(B, T, -1, d_a)  # B, T, N+1, d
        # Mask SSM pipeline
        x = torch.cat((x_v, x_a), dim=2)  # B, T, N+N_a, d
        B, T, N_total, D = x.shape
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
    

    def forward(self, x_v: Tensor, x_a: Tensor, mask=None, mask_ratio = 0.75) -> dict:
        """
        Args:
            x_v (Tensor): Video input of shape [B, C, T, H, W].
            x_a (Tensor): Audio input of shape [B, num_bins, T].
            mask (Tensor, optional): Mask of shape [B, T*(1+N_v+2+N_a+1)], where N_v is the number of video patches and N_a is the number of audio patches.
                ensuring that the cls token and global token are always visible.
                one represent invisible, zero represent visible.
        Returns:
            feat_v (Tensor): Video features of shape [B, K, N_v+1, C_d_clip], where K is the number of layers to return. K=1 as default
            feat_a (Tensor): Audio features of shape [B, K, N_a+1, C_d_clap].
            glbal_feat_v (Tensor): Global video features of shape [B, T, C_d_clip].
            glbal_feat_a (Tensor): Global audio features of shape [B, T, C_d_clap].                               
        """
        B, N_m = mask.shape
        x_vis = self.forward_features(x_v, x_a, mask)     # Student features， B, T*(1+N_v+N_a), d, (1, 1, 550, 768)
        K, B, N_vis, D = x_vis.shape  # K represent the number of student layers to align with CLIP
        T = self.num_frames
        mask = mask.reshape(B, T, -1)
        mask_v = mask[:, :, 1:self.num_patches_v + 1]   # B, T, N_v (1, 10, 196)
        mask_a = mask[:, :, self.num_patches_v + 3:-1]  # B, T, N_a



        num_mask_v = int(mask_v.sum(dim=-1)[0, 0])
        num_mask_a = int(mask_a.sum(dim=-1)[0, 0])
        num_visible_v_per_frame = self.num_patches_v - num_mask_v
        num_visible_a_per_frame = self.num_patches_a - num_mask_a
        
        x_vis = x_vis.reshape(K*B, T, -1, self.embed_dim)  # B, T, N+1, d
        v_vis = x_vis[:, :, 1:num_visible_v_per_frame+1]   # B, T, 49, 768
        a_vis = x_vis[:, :, num_visible_v_per_frame+3:-1]
        global_v = x_vis[:, :, num_visible_v_per_frame+1]  # B, T, 1
        global_a = x_vis[:, :, -1]  # B, T, 1

        _, T, N_vv, C_CLIP = v_vis.shape
        _, T, N_va, C_CLAP = a_vis.shape
        v_vis = v_vis.reshape(B, -1, C_CLIP)  # B, T*N_vv, C_CLIP
        a_vis = a_vis.reshape(B, -1, C_CLAP)
        # align CLIP and ClAP
        # K, B, _, C_CLIP = x_vis.shape              # K represent the number of student layers to align with CLIP
        expand_clip_pos_embed = self.clip_pos_embed.repeat(B, 1, 1).type_as(x_v).to(x_v.device).clone().detach()
        mask_v = mask_v.flatten(1)
        mask_a = mask_a.flatten(1)
        clip_pos_embed_vis = expand_clip_pos_embed[~mask_v].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
        x_clip_full = v_vis + clip_pos_embed_vis # [K, B, N, C_d_clip]

        expand_clap_pos_embed = self.clap_pos_embed.repeat(B, 1, 1).type_as(x_v).to(x_v.device).clone().detach()
        clap_pos_embed_vis = expand_clap_pos_embed[~mask_a].view(B, -1, C_CLAP).unsqueeze(0).repeat(K, 1, 1, 1)
        x_clap_full = a_vis + clap_pos_embed_vis # [K, B, N, C_d_clap]


        x_clip = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip.append(clip_decoder(x_clip_full[idx]))
        x_clip = torch.stack(x_clip) # align and normalize

        x_clap = []
        for idx, clap_decoder in enumerate(self.clap_decoder):
            x_clap.append(clap_decoder(x_clap_full[idx]))
        x_clap = torch.stack(x_clap) # align and normalize
        

        return x_clip, x_clap, global_v, global_a


class CrossMambaFT(CrossMamba): 
    """
    CrossMamba model for fine-tuning with classification heads.
    
    This class inherits from CrossMamba and adds classification heads for both
    video and audio modalities. It removes the CLIP/CLAP decoder components
    that are only needed for pre-training.
    """
    def __init__(
            self,
            # Classification parameters
            num_classes=700,
            fc_drop_rate=0.,
            # Inherit all other parameters from parent class
            **kwargs
    ):
        # Remove CLIP/CLAP decoder parameters that are not needed for fine-tuning
        clip_params_to_remove = [
            'clip_decoder_embed_dim',
            'clap_output_dim', 
            'clip_output_dim',
            'clip_norm_type',
            'clip_return_layer',
            'clip_student_return_interval'
        ]
        
        # Filter out CLIP/CLAP parameters from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in clip_params_to_remove}
        
        # Call parent constructor without CLIP/CLAP parameters
        super().__init__(**filtered_kwargs)
        
        # Remove CLIP/CLAP decoders that are not needed for fine-tuning
        if hasattr(self, 'clip_decoder'):
            delattr(self, 'clip_decoder')
        if hasattr(self, 'clap_decoder'):
            delattr(self, 'clap_decoder')
        if hasattr(self, 'clip_pos_embed'):
            delattr(self, 'clip_pos_embed')
        if hasattr(self, 'clap_pos_embed'):
            delattr(self, 'clap_pos_embed')
        if hasattr(self, 'return_index'):
            delattr(self, 'return_index')
        
        # Add classification heads
        self.num_classes = num_classes
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        # self.head_v = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_a = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

        # Initialize classification heads
        self.head.apply(segm_init_weights)
        
        # 确保LayerNorm和Linear的初始化正确
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)  


    def forward_features(self, x_v: Tensor, x_a: Tensor) -> Tensor:
        """
        Forward pass for feature extraction (no masking for fine-tuning).
        
        Args:
            x_v (Tensor): Video input of shape [B, C, T, H, W].
            x_a (Tensor): Audio input of shape [B, num_bins, T].
        
        Returns:
            Tensor: Combined features of shape [B, N_total, D].
        """
        B, C, T, H, W = x_v.shape

        # Patch embedding - reuse parent class logic
        x_v = self.patch_embed_v(x_v)       # B, C_o, T_o, H_o, W_o
        B, C, T, H, W = x_v.shape
        x_v = x_v.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x_a = x_a.unsqueeze(1)              # B, 1, num_bins, T
        x_a = x_a.transpose(2, 3)           # B, 1, T, num_bins
        x_a = self.patch_embed_a(x_a)       # B, N_a, d_a
        B, N_a, d_a = x_a.shape
        x_a = x_a.reshape(B*T, -1, d_a)     # B*T, N_a // T, d_a
    
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
        x_v = x_v.reshape(B*T, N_v, D)       # B*T, N_v, d
        x_a = x_a.reshape(B*T, N_a, d_a)     # B*T, N_a, d_a

        # Add cls token
        cls_tokens_v = self.cls_token_v.expand(B*T, -1, -1)
        cls_tokens_a = self.cls_token_a.expand(B*T, -1, -1)
        x_v = torch.cat((cls_tokens_v, x_v), dim=1)  # B*T, N_v+1, d
        x_a = torch.cat((cls_tokens_a, x_a), dim=1)  # B*T, N_a+1, d_a

        # Add Global token (if not using global pooling)
        if not self.use_global_pooling:
            global_token_v = self.global_token_v.expand(B*T, -1, -1)
            global_token_a = self.global_token_a.expand(B*T, -1, -1)
            x_v = torch.cat((x_v, global_token_v), dim=1)   # add global token at tail
            x_a = torch.cat((x_a, global_token_a), dim=1)

        x_v, x_a = x_v.reshape(B, T, -1, D), x_a.reshape(B, T, -1, d_a)  # B, T, N+1, d
        
        # Concatenate video and audio features (no masking for fine-tuning)
        x = torch.cat((x_v, x_a), dim=2)  # B, T, N_v+N_a+special_tokens, d
        B, T, N_total, D = x.shape
        x = x.reshape(B, -1, D)  # B, T*N_total, d

        # Mamba layers
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )

        # Final normalization
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

        return hidden_states  # B, T*N_total, D

    def forward(self, x_v: Tensor, x_a: Tensor) -> dict:
        """
        Full forward pass with classification outputs.
        
        Args:
            x_v (Tensor): Video input of shape [B, C, T, H, W].
            x_a (Tensor): Audio input of shape [B, num_bins, T].
        
        Returns:
            dict: Dictionary containing classification logits for video and audio.
        """
        features = self.forward_features(x_v, x_a)  # B, T*N_total, D
        B, _, D = features.shape
        T = self.num_frames
        
        # Reshape to separate temporal and spatial dimensions
        features = features.reshape(B, T, -1, D)  # B, T, N_total, D
        
        # Extract CLS tokens for each modality
        if self.use_global_pooling:
            # If using global pooling, apply pooling to get global representations
            # We need to extract video and audio patches separately
            N_v = self.num_patches_v + 1  # +1 for cls token
            N_a = self.num_patches_a + 1  # +1 for cls token
            
            video_features = features[:, :, :N_v, :]  # B, T, N_v+1, D
            audio_features = features[:, :, N_v:N_v+N_a, :]  # B, T, N_a+1, D
            
            # Apply global pooling
            video_features = video_features.reshape(B*T, N_v, D)
            audio_features = audio_features.reshape(B*T, N_a, D)
            
            cls_v = self.global_pooling_head_v(video_features)  # B*T, D
            cls_a = self.global_pooling_head_a(audio_features)  # B*T, D
            
            cls_v = cls_v.reshape(B, T, D).mean(dim=1)  # B, D - average over time
            cls_a = cls_a.reshape(B, T, D).mean(dim=1)  # B, D - average over time
        else:
            # Use global tokens (last tokens for each modality)
            N_v = self.num_patches_v + 2  # +1 for cls, +1 for global
            # Use all tokens for classification:
            global_v = features[:, :, :N_v, :]  # B, T, num_patches_v+2, D
            global_a = features[:, :, N_v:, :]   # B, T, num_patches_v_a+2, D
            global_feat = torch.concatenate((global_v, global_a), dim=-2)  # B, T, N_total, D
            global_feat = global_feat.mean(dim=1)  # B, N_total, D

        head_input = global_feat.mean(dim=1)  # B, D
        logits = self.head_drop(head_input)  # Apply dropout
        logits = self.head(logits)  # Final classification layer

        return {
            'logits': logits,
            'feat_v': global_v,
            'feat_a': global_a,
        }


class UniModalMamba(nn.Module):
    """
    UniModalMamba is a single-modality version of CrossMamba that processes
    only one modality (video or audio) with mask+decoder reconstruction pipeline.
    
    Args:
        modality (str): Either 'video' or 'audio' to specify which modality to process
        img_size (int): Input image size for video, or treated as patch size for audio
        audio_length (int): Length of audio sequence (only used for audio modality)  
        num_bins (int): Number of frequency bins for audio (only used for audio modality)
        patch_size (int): Patch size for vision transformer
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer layers
        num_frames (int): Number of frames in video sequence
        clip_decoder_embed_dim (int): Decoder embedding dimension 
        clip_output_dim (int): Output dimension for decoder
        clip_return_layer (int): Number of layers to return for decoding
        clip_student_return_interval (int): Interval between returned layers
        clip_norm_type (str): Normalization type for decoder
        use_global_pooling (bool): Whether to use global pooling
    """
    def __init__(
            self,
            modality='audio',  # 'video' or 'audio'
            img_size=224,
            audio_length=1024,
            num_bins=64,
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
            # Model general parameters
            depth=32,
            drop_path_rate=0.4,
            num_frames=16,
            device=None,
            dtype=None,
            # Decoder parameters for mask+reconstruction
            clip_decoder_embed_dim=768,
            clip_output_dim=768,
            clip_return_layer=1,
            clip_student_return_interval=1,
            clip_norm_type='l2',
            use_global_pooling=False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.modality = modality
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.use_global_pooling = use_global_pooling
        
        # Single modality patch embedding
        if modality == 'video':
            self.patch_embed = PatchEmbedVideo(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                in_chans=3,
                kernel_size=kernel_size,
            )
            self.num_patches = int(img_size * img_size / (patch_size * patch_size))  # 196 for 224x224 with 16x16 patches
        elif modality == 'audio':
            self.patch_embed = PatchEmbedImg(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                in_chans=1,
            )
            # For audio: calculate patches based on audio_length and num_bins
            self.patch_embed.num_patches = int(audio_length * num_bins / 256)
            self.num_patches = self.patch_embed.num_patches // num_frames
        else:
            raise ValueError(f"Unsupported modality: {modality}. Must be 'video' or 'audio'")

        # Mamba layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.bimamba = bimamba
        self.layers = nn.ModuleList([
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
            )
            for i in range(depth)
        ])
        
        self.depth = depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Tokens and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if not use_global_pooling:
            self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        
        # Initialize position embeddings
        pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)
        self.pos_embed.data.copy_(pos_embed.float())
        self.temporal_pos_embed.data.copy_(torch.zeros(1, num_frames // kernel_size, embed_dim).float())
        
        # Global pooling head (if used)
        if use_global_pooling:
            self.global_pooling_head = GlobalPoolingHead(embed_dim)

        # Output normalization
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        
        # Decoder for mask+reconstruction pipeline (similar to CLIP/CLAP decoder)
        self.decoder = nn.ModuleList([
            Linear_Decoder(
                output_dim=clip_output_dim, 
                embed_dim=clip_decoder_embed_dim, 
                norm_layer=nn.LayerNorm, 
                clip_norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])
        
        # Position embedding for decoder
        self.decoder_pos_embed = get_sinusoid_encoding_table(
            self.num_patches * num_frames // kernel_size, 
            clip_decoder_embed_dim
        )
        
        # Return layer indices for decoder
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        
        # Initialize parameters
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if not use_global_pooling:
            trunc_normal_(self.global_token, std=.02)
        
        # Mamba initialization
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward_features(self, x: Tensor, mask=None) -> dict:
        """
        Forward pass for feature extraction with mask support.
        
        Args:
            x (Tensor): Input tensor
                - For video: [B, C, T, H, W]
                - For audio: [B, num_bins, T] or [B, C, T] 
            mask (Tensor, optional): Mask tensor of shape [B, T*(1+N_patches+1)] for masking
                True values are masked (invisible), False values are visible
        
        Returns:
            Tensor: Features from visible patches for decoder reconstruction
        """
        if self.modality == 'video':
            B, C, T, H, W = x.shape
            # Video patch embedding
            x = self.patch_embed(x)  # B, C_o, T_o, H_o, W_o
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)  # B*T, N_patches, D
        else:  # audio
            B = x.shape[0]
            T = self.num_frames
            if len(x.shape) == 3:  # B, num_bins, T
                x = x.unsqueeze(1)  # B, 1, num_bins, T
            x = x.transpose(2, 3)  # B, 1, T, num_bins
            x = self.patch_embed(x)  # B, N_total, D
            B, N_total, D = x.shape
            x = x.reshape(B*T, -1, D)  # B*T, N_patches_per_frame, D
        
        # Add spatial position embedding
        pos_embed = self.pos_embed.expand(B*T, -1, -1)
        x = x + pos_embed
        
        # Add temporal position embedding
        x = x.reshape(B, T, -1, x.shape[-1])  # B, T, N_patches, D
        B, T, N_patches, D = x.shape
        temporal_pos_embed = self.temporal_pos_embed.unsqueeze(-2).expand(B, -1, N_patches, -1)
        x = x + temporal_pos_embed
        x = x.reshape(B*T, N_patches, D)  # B*T, N_patches, D
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B*T, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # B*T, N_patches+1, D
        
        # Add global token (if not using global pooling)
        if not self.use_global_pooling:
            global_tokens = self.global_token.expand(B*T, -1, -1)
            x = torch.cat((x, global_tokens), dim=1)  # B*T, N_patches+2, D
        
        # Reshape for sequence processing
        x = x.reshape(B, -1, D)  # B, T*(N_patches+tokens), D
        
        # Apply mask if provided
        if mask is not None:
            x_vis = x[~mask].reshape(B, -1, D)  # Only visible patches
        else:
            x_vis = x  # No masking
        
        # Store features from different layers for decoder
        x_features = []
        
        # Pass through Mamba layers
        residual = None
        hidden_states = x_vis
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
            # Store features from specified layers for decoder
            if (idx - 1) in self.return_index:
                x_features.append(self.norm(residual.to(dtype=self.norm.weight.dtype)))
        
        # Final normalization
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
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
        
        # Add final layer if in return index
        if (self.depth - 1) in self.return_index:
            x_features.append(hidden_states)
        
        x_features = torch.stack(x_features) if x_features else hidden_states.unsqueeze(0)
        
        return x_features

    def forward(self, x: Tensor, mask=None) -> dict:
        """
        Full forward pass with mask+decoder reconstruction pipeline.
        
        Args:
            x (Tensor): Input tensor
                - For video: [B, C, T, H, W]  
                - For audio: [B, num_bins, T] or [B, C, T]
            mask (Tensor, optional): Mask tensor for masking patches
                True values are masked (invisible), False values are visible
        
        Returns:
            dict: Dictionary containing:
                - 'features': Reconstructed features from decoder
                - 'global_features': Global features for the modality
        """
        # Get features from encoder
        x_vis = self.forward_features(x, mask)  # K, B, N_vis, D
        K, B, N_vis, D = x_vis.shape
        T = self.num_frames
        
        # Extract visible patches and global features
        if mask is not None:
            # Calculate number of visible patches per frame
            if self.use_global_pooling:
                num_tokens = 1  # only CLS token
            else:
                num_tokens = 2  # CLS + global token
            
            # Reshape mask to separate by frame
            mask = mask.reshape(B, T, -1)
            patch_mask = mask[:, :, 1:-num_tokens+1] if num_tokens > 1 else mask[:, :, 1:]  # B, T, N_patches
            
            num_visible_per_frame = self.num_patches - int(patch_mask.sum(dim=-1)[0, 0])
            
            # Extract visible patches and global features
            x_vis = x_vis.reshape(K*B, T, -1, D)  # K*B, T, N_vis_total, D
            patches_vis = x_vis[:, :, 1:num_visible_per_frame+1]  # K*B, T, N_vis_patches, D
            
            if not self.use_global_pooling:
                global_feat = x_vis[:, :, -1]  # K*B, T, D (global token)
            else:
                # For global pooling, we need to pool over visible patches
                global_feat = patches_vis.mean(dim=2)  # K*B, T, D
            
            # Reshape for decoder
            patches_vis = patches_vis.reshape(B, -1, D)  # B, T*N_vis_patches, D
            
            # Add position embedding for visible patches
            expand_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
            patch_mask_flat = patch_mask.flatten(1)  # B, T*N_patches
            pos_embed_vis = expand_pos_embed[~patch_mask_flat].view(B, -1, D).unsqueeze(0).repeat(K, 1, 1, 1)
            x_decoder_input = patches_vis + pos_embed_vis  # K, B, N_vis, D
            
        else:
            # No masking - use all patches
            x_vis = x_vis.reshape(K*B, T, -1, D)
            patches_vis = x_vis[:, :, 1:-1 if not self.use_global_pooling else 1:]  # K*B, T, N_patches, D
            
            global_feat = x_vis[:, :, -1] 
            
            patches_vis = patches_vis.reshape(B, -1, D)  # B, T*N_patches, D
            x_decoder_input = patches_vis.unsqueeze(0).repeat(K, 1, 1, 1)  # K, B, T*N_patches, D
        
        # Apply decoder to reconstruct features
        x_decoded = []
        for idx, decoder in enumerate(self.decoder):
            x_decoded.append(decoder(x_decoder_input[idx]))
        x_decoded = torch.stack(x_decoded)  # K, B, N_vis, output_dim
        
        # Reshape global features
        global_feat = global_feat.reshape(B, T, D)  # B, T, D
        
        return {
            'features': x_decoded,
            'global_features': global_feat,
        }

class UniModalMamba_FT(UniModalMamba):
    def __init__(
        self,
        modality='audio',
        num_classes=700,
        fc_drop_rate=0.,
        **kwargs
    ):
        # Remove decoder-related parameters
        clip_params_to_remove = [
            'clip_decoder_embed_dim',
            'clip_output_dim',
            'clip_return_layer',
            'clip_student_return_interval',
            'clip_norm_type',
        ]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in clip_params_to_remove}

        # Initialize base UniModalMamba (without decoder)
        super().__init__(modality=modality, **filtered_kwargs)
        self.return_index = []
        self.return_index.append(self.depth - 1)  # Only return the last layer for fine-tuning
        # Delete decoder components
        if hasattr(self, 'decoder'):
            del self.decoder
        if hasattr(self, 'decoder_pos_embed'):
            del self.decoder_pos_embed
        # Add classification head
        self.num_classes = num_classes
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

        self.head.apply(segm_init_weights)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor) -> Tensor:
        # Use base forward_features without masking
        x_vis = super().forward_features(x, mask=None)  # (K=1, B, N, D)
        x_vis = x_vis.squeeze(0)  # B, N, D
        return x_vis

    def forward(self, x: Tensor) -> dict:
        x_feat = self.forward_features(x)  # B, N, D

        if self.use_global_pooling:
            # Mean pooling across all tokens
            x_global = x_feat.mean(dim=1)
        else:
            # Use last token as global token
            x_global = x_feat[:, -1, :]

        logits = self.head_drop(x_global)
        logits = self.head(logits)

        return {
            'logits': logits,
            'features': x_feat,
            'global_feature': x_global,
        }
            




if __name__ == '__main__':
    # Test Script, run in the root directory of the project
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    from models.mamba_pretrain import CrossMamba, CrossMambaFT, UniModalMamba

    # Test CrossMamba (for pre-training)
    print("=" * 50)
    print("Testing CrossMamba (Pre-training)")
    print("=" * 50)
    model_pretrain = CrossMamba()
    print("CrossMamba model created successfully.")
    print(f"Video patches: {model_pretrain.patch_embed_v.num_patches}")
    print(f"Audio patches: {model_pretrain.patch_embed_a.num_patches}")

    # Test CrossMambaFT (for fine-tuning)
    print("\n" + "=" * 50)
    print("Testing CrossMambaFT (Fine-tuning)")
    print("=" * 50)
    model_ft = CrossMambaFT(num_classes=700, fc_drop_rate=0.1)
    print("CrossMambaFT model created successfully.")
    print(f"Number of classes: {model_ft.num_classes}")

    # Test UniModalMamba (for single modality with mask+decoder)
    print("\n" + "=" * 50)
    print("Testing UniModalMamba (Single Modality with Mask+Decoder)")
    print("=" * 50)
    
    # Test video-only model with mask+decoder
    model_video = UniModalMamba(
        modality='video', 
        num_frames=10,
        clip_decoder_embed_dim=512,
        clip_output_dim=768,
        clip_return_layer=1
    )
    print("UniModalMamba (video) model created successfully.")
    print(f"Video patches: {model_video.num_patches}")
    print(f"Modality: {model_video.modality}")
    print(f"Decoder layers: {len(model_video.decoder)}")
    
    # Test audio-only model with mask+decoder
    model_audio = UniModalMamba(
        modality='audio', 
        num_frames=10, 
        audio_length=960,
        clip_decoder_embed_dim=512,
        clip_output_dim=768,
        clip_return_layer=1
    )
    print("UniModalMamba (audio) model created successfully.")
    print(f"Audio patches: {model_audio.num_patches}")
    print(f"Modality: {model_audio.modality}")
    print(f"Decoder layers: {len(model_audio.decoder)}")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test inputs
    v = torch.randn(1, 3, 10, 224, 224)  # Video input
    a = torch.randn(1, 128, 960)         # Audio input
    
    # Test mask for pre-training
    ones = torch.ones(1, 10)
    mask = torch.cat([
        ones,
        torch.ones(1, 10 * int(14 * 14 * 0.75)),
        torch.zeros(1, 10 * int(14 * 14 * 0.25)),
        ones,
        ones,     
        torch.ones(1, 10 * int(6 * 8 * 0.75)),      # 6 = 960 / 10 / 16
        torch.zeros(1, 10 * int(6 * 8 * 0.25)),
        ones, 
    ], dim=-1).to(torch.bool)

    # Create masks for UniModalMamba
    # Video mask: [CLS] + [patches] + [global] where some patches are masked
    video_mask = torch.cat([
        torch.zeros(1, 1),  # CLS token always visible
        torch.ones(1, 10 * int(14 * 14 * 0.75)),   # 75% patches masked
        torch.zeros(1, 10 * int(14 * 14 * 0.25)),  # 25% patches visible
        torch.zeros(1, 1),  # Global token always visible
    ], dim=-1).to(torch.bool)
    
    # Audio mask: [CLS] + [patches] + [global] where some patches are masked
    audio_mask = torch.cat([
        torch.zeros(1, 1),  # CLS token always visible
        torch.ones(1, 10 * int(6 * 8 * 0.75)),     # 75% patches masked
        torch.zeros(1, 10 * int(6 * 8 * 0.25)),    # 25% patches visible
        torch.zeros(1, 1),  # Global token always visible
    ], dim=-1).to(torch.bool)

    # Move to device
    model_pretrain = model_pretrain.to(device)
    model_ft = model_ft.to(device)
    model_video = model_video.to(device)
    model_audio = model_audio.to(device)
    v = v.to(device)
    a = a.to(device)
    mask = mask.to(device)
    video_mask = video_mask.to(device)
    audio_mask = audio_mask.to(device)

    # Test pre-training model
    print("\nTesting pre-training forward pass...")
    try:
        with torch.no_grad():
            x_clip, x_clap, global_v, global_a = model_pretrain(v, a, mask)
            print(f"CLIP output shape: {x_clip.shape}")
            print(f"CLAP output shape: {x_clap.shape}")
            print(f"Global video shape: {global_v.shape}")
            print(f"Global audio shape: {global_a.shape}")
    except Exception as e:
        print(f"Pre-training forward pass failed: {e}")

    # Test fine-tuning model
    print("\nTesting fine-tuning forward pass...")
    try:
        with torch.no_grad():
            outputs = model_ft(v, a)
            print(f"Logits shape: {outputs['logits'].shape}")
            print(f"Video features shape: {outputs['feat_v'].shape}")
            print(f"Audio features shape: {outputs['feat_a'].shape}")
    except Exception as e:
        print(f"Fine-tuning forward pass failed: {e}")

    # Test UniModalMamba video model with mask+decoder
    print("\nTesting UniModalMamba (video) mask+decoder forward pass...")
    try:
        with torch.no_grad():
            outputs_video = model_video(v, video_mask)
            print(f"Video decoded features shape: {outputs_video['features'].shape}")
            print(f"Video global features shape: {outputs_video['global_features'].shape}")
    except Exception as e:
        print(f"UniModalMamba (video) forward pass failed: {e}")

    # Test UniModalMamba audio model with mask+decoder
    print("\nTesting UniModalMamba (audio) mask+decoder forward pass...")
    try:
        with torch.no_grad():
            outputs_audio = model_audio(a, audio_mask)
            print(f"Audio decoded features shape: {outputs_audio['features'].shape}")
            print(f"Audio global features shape: {outputs_audio['global_features'].shape}")
    except Exception as e:
        print(f"UniModalMamba (audio) forward pass failed: {e}")

    # Test UniModalMamba without mask (for comparison)
    print("\nTesting UniModalMamba (video) without mask...")
    try:
        with torch.no_grad():
            outputs_video_no_mask = model_video(v, None)
            print(f"Video decoded features shape (no mask): {outputs_video_no_mask['features'].shape}")
            print(f"Video global features shape (no mask): {outputs_video_no_mask['global_features'].shape}")
    except Exception as e:
        print(f"UniModalMamba (video) without mask failed: {e}")

    print("\nAll tests completed!")
    print("UniModalMamba now supports mask+decoder reconstruction pipeline like CrossMamba!")