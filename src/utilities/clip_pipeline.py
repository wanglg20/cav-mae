import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor
from sklearn.metrics.pairwise import cosine_similarity

# 选择设备：CUDA 或 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型和预处理器
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# 加载 CLAP 模型和预处理器
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# 测试文本
text = "a photo of a cat"
text_2 = "sound of a cat meowing"

# 处理 CLIP 文本
clip_inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    clip_features = clip_model.get_text_features(**clip_inputs)

# 处理 CLAP 文本
clap_inputs = clap_processor(text=text_2, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    clap_features = clap_model.get_text_features(**clap_inputs)

print("CLIP Features Shape:", clip_features.shape)
print("CLAP Features Shape:", clap_features.shape)
# 归一化特征
clip_features = clip_features / clip_features.norm(p=2, dim=-1, keepdim=True)
clap_features = clap_features / clap_features.norm(p=2, dim=-1, keepdim=True)

# 计算余弦相似度
cos_sim = cosine_similarity(clip_features.cpu().numpy(), clap_features.cpu().numpy())

# 输出余弦相似度
print("Cosine Similarity between CLIP and CLAP for the same text:", cos_sim[0][0])
