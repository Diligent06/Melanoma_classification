import torch
from dataloader import MedicalDataset
from torch.utils.data import Dataset, DataLoader
import torch
from model.model import Backbone_MoE
import torch.nn as nn
from tqdm import tqdm
import os
import torchvision.transforms as T
from os.path import join
from PIL import Image
from torchvision.utils import save_image
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32


model = Backbone_MoE(expert_num=4, device=device, dtype=dtype)
model.to(device=device, dtype=dtype)



val_dataset = MedicalDataset(csv_path='../datasets/SkinCON/data/train.csv')

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1
)

for batch in tqdm(val_loader):
    images = batch[0].to(device, dtype=dtype)
    
    seg_mask = model.segment_forward(images)

    seg_mask = seg_mask[0]
    mask = seg_mask.detach().cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    
    img = Image.fromarray(mask)
    img.save('test_mask.png')
    
    save_image(images,  "test_rgb.png", normalize=True)
    
    from IPython import embed; embed()


