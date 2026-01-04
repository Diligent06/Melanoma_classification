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


checkpoint_path = './checkpoints/best_model_epoch_831.pth'
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

model.seg_model_init()

val_dataset = MedicalDataset(csv_path='../datasets/SkinCON/data/train.csv')

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

real_label_id = []
i = 0
with torch.no_grad():
    for batch in tqdm(val_loader):
        if len(real_label_id) >=4:
            break
        if batch[2][0] in real_label_id:
            continue
        
        
        images = batch[0].to(device, dtype=dtype)
        value_tensor = batch[1].to(device, dtype=dtype)
        final_label_id = batch[2].to(device, dtype=torch.long)
        sex_id = batch[3].to(device, dtype=dtype)
        anatom_site_general_id = batch[4].to(device, dtype=dtype)
        age = batch[5].to(device, dtype=dtype)

        context_features = {'age': age, 'sex': sex_id, 'anatom_site_general': anatom_site_general_id}
        prob, out, seg_mask = model.forward_eval(images, context_features)

        
        seg_mask = seg_mask[0]
        
        mask = seg_mask.detach().cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        
        img = Image.fromarray(mask)
        img.save(join('./temp_output_folder', f'gray_{i}.png'))
        
        real_id = batch[2][0]
        predict_id = out.argmax(dim=1)[0]
        
        if real_id != predict_id:
            continue
        
        
        print(value_tensor)
        print(final_label_id)
        torch.save(value_tensor, join('./temp_output_folder', f"gt_prob_{i}.pt"))

        save_image(images,  join('./temp_output_folder', f"rgb_real{real_id}_predict{predict_id}_{i}.png"), normalize=True)
        
        torch.save(prob, join('./temp_output_folder', f"prob_{i}.pt"))
        # torch.save(out, f"out_{i}.pt")
        # torch.save(final_label_id, f"final_label_id_{i}.pt")
        i += 1
        real_label_id.append(batch[2][0])
        
        # from IPython import embed; embed()





