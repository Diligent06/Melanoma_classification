from model.model import Backbone_MoE
import torch
from PIL import Image
from torchvision import transforms




image_path = '../datasets/isic_2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000012.jpg'
img = Image.open(image_path).convert('RGB')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

x = transform(img).unsqueeze(0)  # Add batch dimension
x = torch.cat([x, x], dim=0)  # batch size = 2  


model = Backbone_MoE()

model.training = True

context_features = [{"age": 55.0, "sex": 2, "anatom_site_general": 3}, 
                    {"age": 51.0, "sex": 0, "anatom_site_general": 8}]
out, gate_w = model.forward(x, context_features)

from IPython import embed; embed()


