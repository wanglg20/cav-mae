import torch
from torch import nn
from typing import Optional, Union, Tuple
from transformers.models.clap.modeling_clap import ClapAudioModelOutput, ClapAudioPatchEmbed, ClapAudioStage, ClapAudioPatchMerging
from transformers.models.clap.modeling_clap import ClapAudioModel
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

clap_encoder = clap_model.audio_model
weight_path = '/data/wanglinge/project/cav-mae/src/weight/teacher/clap.pth'
clap_encoder.load_state_dict(torch.load(weight_path, map_location=device), strict=True)

audio = torch.randn(1, 1, 96, 64).to(device)  # 假设的输入音频特征

# out1 = clap_model.get_audio_features(audio, is_longer=torch.tensor([1]).bool().to(device))
# print(out1.shape)  # 输出音频特征的形状 # 1, 512
outputs = clap_encoder(audio, is_longer=torch.tensor([1]).bool().to(device))
print(outputs.last_hidden_state.shape)  # 1, 768, 2, 32， 



# class ClapAudioEncoder(nn.Module):
#     def __init__(self, config, audio_model=None):
#         super().__init__()
        
#         self.num_layers = len(config.depths)
#         self.config = config
#         self.patch_embed = ClapAudioPatchEmbed(config)
#         self.enable_fusion = config.enable_fusion
#         self.spec_size = config.spec_size
#         self.freq_ratio = config.spec_size // config.num_mel_bins
#         self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))

#         # Initialize layers and configurations as in the original ClapAudioEncoder
#         drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu")]
#         grid_size = self.patch_embed.grid_size
#         self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]
        
#         self.layers = nn.ModuleList(
#             [
#                 ClapAudioStage(
#                     config=config,
#                     dim=int(config.patch_embeds_hidden_size * 2**i_layer),
#                     input_resolution=self.input_resolutions[i_layer],
#                     depth=config.depths[i_layer],
#                     num_heads=config.num_attention_heads[i_layer],
#                     drop_path=drop_path_rate[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
#                     downsample=ClapAudioPatchMerging if (i_layer < self.num_layers - 1) else None,
#                 )
#                 for i_layer in range(self.num_layers)
#             ]
#         )

#         # Batch norm and layer norm for audio features
#         self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
#         self.norm = nn.LayerNorm(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)

#         # New audio model integration
#         self.audio_model = audio_model  # This could be a separate pre-trained audio feature extractor

#     def reshape_mel2img(self, normalized_input_features):
#         # Reshapes the spectrogram to the common image-like format
#         _, _, time_length, freq_length = normalized_input_features.shape
#         spec_width = int(self.spec_size * self.freq_ratio)
#         spec_height = self.spec_size // self.freq_ratio

#         # Resizing to match the expected shape if necessary
#         if time_length < spec_width:
#             normalized_input_features = nn.functional.interpolate(
#                 normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
#             )
#         if freq_length < spec_height:
#             normalized_input_features = nn.functional.interpolate(
#                 normalized_input_features, (time_length, spec_height), mode="bicubic", align_corners=True
#             )

#         # Reshaping to prepare for audio processing
#         batch, channels, time, freq = normalized_input_features.shape
#         normalized_input_features = normalized_input_features.reshape(batch, channels * self.freq_ratio, time // self.freq_ratio, freq)
#         normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
#         normalized_input_features = normalized_input_features.reshape(batch, channels, freq * self.freq_ratio, time // self.freq_ratio)

#         return normalized_input_features

#     def forward(
#         self,
#         input_features,
#         is_longer: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = False,
#         return_dict: Optional[bool] = True,
#     ) -> Union[Tuple, ClapAudioModelOutput]:
        
#         input_features = input_features.transpose(1, 3)
#         normalized_input_features = self.batch_norm(input_features)
#         normalized_input_features = normalized_input_features.transpose(1, 3)
#         audio_features = normalized_input_features
#         # Further processing of audio features through the encoder layers
#         hidden_states = self.reshape_mel2img(audio_features)

#         all_hidden_states = () if not output_attentions else None

#         # Process through all layers
#         for i, layer_module in enumerate(self.layers):
#             layer_outputs = layer_module(hidden_states)
#             hidden_states = layer_outputs[0]  # Update hidden states

#         last_hidden_state = self.norm(hidden_states)

#         # Reshape the last hidden state and apply pooling
#         batch_size, _, n_channels = last_hidden_state.shape
#         latent_output = self.avgpool(last_hidden_state.flatten(2)).flatten(1)

#         if not return_dict:
#             return last_hidden_state, latent_output

#         return ClapAudioModelOutput(
#             last_hidden_state=last_hidden_state,
#             pooler_output=latent_output
#         )

# class ClapAudioConfig:
#     def __init__(self):
#         # 输入特征相关
#         self.spec_size = 224  # 输入声谱图的尺寸（宽度和高度）
#         self.num_mel_bins = 128  # Mel 频率尺度的频率桶数（音频特征）
#         self.patch_embeds_hidden_size = 768  # 每个 patch 的隐藏大小
#         self.enable_fusion = True  # 是否启用融合
#         self.patch_stride = (4, 4)  # 每个 patch 的步幅

#         # Transformer 相关
#         self.depths = [2, 2, 6, 2]  # 每一层 Transformer 的深度
#         self.num_attention_heads = [12, 12, 12, 12]  # 每一层 Transformer 中的注意力头数
#         self.drop_path_rate = 0.1  # 随机丢弃路径率（drop-path rate）

#         # 用于网络结构的配置
#         self.grid_size = (14, 14)  # 每层网络的格子大小
#         self.input_resolutions = [(224, 224), (112, 112), (56, 56), (28, 28)]  # 各层输入分辨率

#         # 其他配置
#         self.freq_ratio = 2  # 频率分辨率的比率
#         self.num_layers = len(self.depths)  # Transformer 总层数

#         # 可选的音频模型相关配置
#         self.use_audio_model = True  # 是否启用音频模型
#         self.audio_embedding_dim = 512  # 音频模型的嵌入维度（可以与视觉模型共享维度）

# # 创建配置对象
# config = ClapAudioConfig()

# audio_model = ClapAudioModel(config)

# # 创建 ClapAudioEncoder
# clap_audio_encoder = ClapAudioEncoder(config, audio_model=audio_model)

# # 输入音频特征（例如，声谱图）
# input_features = torch.randn(batch_size, 1, 1000, 128)  # 假设的输入

# # 调用模型进行前向推理
# outputs = clap_audio_encoder(input_features)

# # 输出特征
# print("Last hidden state:", outputs.last_hidden_state.shape)
# print("Latent output:", outputs.pooler_output.shape)       