import os
import sys
sys.path.append(os.getcwd())
from MobileUNETR.architectures.mobileunetr import build_mobileunetr_s, build_mobileunetr_xs, build_mobileunetr_xxs
import torch
from PIL import Image
from torchvision import transforms
from os.path import join
from torchvision.utils import save_image

mobileunetr_xxs = build_mobileunetr_xxs(num_classes=1, image_size=512)

ckpt = torch.load('/home/diligent/workspace/Medical/ckpts/final_model_files/isic_2018_pytorch_model.bin', map_location='cpu')

# from IPython import embed; embed()

mobileunetr_xxs.load_state_dict(ckpt)

image_paths = ['rgb_real1_predict1_0.png', 'rgb_real2_predict2_1.png', 'rgb_real3_predict3_3.png', 'rgb_real6_predict6_2.png']
for i in range(4):
    image_path = join('/home/diligent/workspace/Medical/temp_output_folder', image_paths[i])
    img = Image.open(image_path).convert('RGB')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    # ])
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)
    # x = transform(img).unsqueeze(0)  # Add batch dimension

    x = img_tensor.unsqueeze(0)

    # from IPython import embed; embed()

    # data = torch.randn((4, 3, 512, 512))
    out = mobileunetr_xxs.forward(x)

    out = out[0, 0]
    binary = (out > 0).float()
    binary_image = (binary * 255).byte().cpu().numpy()

    img = Image.fromarray(binary_image)
    img.save(join('./temp_output_folder', f'gray_{i}.png'))





