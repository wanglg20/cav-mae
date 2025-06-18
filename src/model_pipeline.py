import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from models.mamba_pretrain import CrossMamba

model = CrossMamba()
print("MambaPretrain model created successfully.")
print(model.patch_embed_v.num_patches, model.patch_embed_a.num_patches)

import torch
v = torch.randn(1, 16, 3, 224, 224)  # Video input

