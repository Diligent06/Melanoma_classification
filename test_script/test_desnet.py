import torch
from torchvision import models
import torch.nn.functional as F

model = models.densenet201(
    weights=models.DenseNet201_Weights.IMAGENET1K_V1
)
model.eval()

x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    features = model.features(x)   # [1, 1920, 7, 7]
    features = F.relu(features)
    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    pooled = pooled.view(pooled.size(0), -1)  # [1, 1920]

print(pooled.shape)


