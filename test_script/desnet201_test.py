import torch
from torchvision import models

# 1️⃣ Load pretrained DenseNet-201
model = models.densenet201(
    weights=models.DenseNet201_Weights.IMAGENET1K_V1
)
model.eval()  # inference mode

# 2️⃣ Generate random input (batch_size=2, RGB, 224x224)
x = torch.randn(2, 3, 224, 224)

# 3️⃣ Forward pass
with torch.no_grad():
    logits = model(x)

print("Output shape:", logits.shape)
