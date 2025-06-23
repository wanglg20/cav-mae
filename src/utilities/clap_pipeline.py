
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor
import torchaudio

# 设置音频后端为 sox

wav_path = '/data/wanglinge/project/cav-mae/src/data/k700/test/audio/4mQbPntWUZA_000015_000025.wav'

clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
clap_input = clap_processor(audios = waveform, return_tensors="pt", padding=True)
print(clap_input.shape)
# clap_input = clap_processor(text=text_2, return_tensors="pt", padding=True).to(device)