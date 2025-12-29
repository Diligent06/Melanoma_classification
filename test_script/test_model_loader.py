from MobileUNETR.architectures.mobileunetr import build_mobileunetr_s, build_mobileunetr_xs, build_mobileunetr_xxs
import torch
from PIL import Image
from torchvision import transforms


mobileunetr_xxs = build_mobileunetr_xxs(num_classes=1, image_size=512)

ckpt = torch.load('/home/diligent/workspace/Medical/ckpts/final_model_files/isic_2018_pytorch_model.bin', map_location='cpu')

# from IPython import embed; embed()

mobileunetr_xxs.load_state_dict(ckpt)

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


# from IPython import embed; embed()

# data = torch.randn((4, 3, 512, 512))
out = mobileunetr_xxs.forward(x)

out = out[0, 0]
binary = (out > 0).float()
binary_image = (binary * 255).byte().cpu().numpy()

img = Image.fromarray(binary_image)
img.save('segmentation_output_1.png')





