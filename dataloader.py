import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import json
from os.path import join

class MedicalDataset(Dataset):
    def __init__(self, csv_path):
        """
        csv_path: 数据 csv，至少包含 columns ['image_name', 'age', 'sex', 'anatom_site_general', 'label']
        img_dir: 图片所在文件夹
        transform: torchvision transform
        """
        self.csv_path = csv_path 
        self.df = pd.read_csv(csv_path)
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.img_dir = '../datasets/isic_2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/'
        self.transform = T.Compose([
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ])

        self.sex2id = json.load(open('../sex_dict.json', 'r'))
        self.anatom_site_general2id = json.load(open('../anatom_site_general_dict.json', 'r'))
        self.label_dict = json.load(open('../label_dict.json', 'r'))
        
        self.sex2id = {v: int(k) for k, v in self.sex2id.items()}
        self.anatom_site_general2id = {v: int(k) for k, v in self.anatom_site_general2id.items()}
        
        self.skincon_path = '../datasets/SkinCON/SkinCON.csv'
        self.skincon_df = pd.read_csv(self.skincon_path)
        
        self.isic_2019_meta_path = '../datasets/isic_2019/ISIC_2019_Training_Metadata.csv'
        self.isic_2019_meta = pd.read_csv(self.isic_2019_meta_path)
        
        # from IPython import embed; embed()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx > len(self.df):
            raise IndexError
        row = self.df.iloc[idx]
        image_name = row['filename']
        label = row['labels']
        skincon_row = self.skincon_df[self.skincon_df['Filename'] == image_name]
        isic_2019_row = self.isic_2019_meta[self.isic_2019_meta['image'] == image_name.split('.')[0]]
        sex = isic_2019_row['sex'].item()
        anatom_site_general = isic_2019_row['anatom_site_general'].item()
        age = isic_2019_row['age_approx'].item()
        
        if pd.isna(sex):
            sex_id = 0
        else:
            sex_id = self.sex2id[sex]
        if pd.isna(anatom_site_general):
            anatom_site_general_id = 0
        else:
            anatom_site_general_id = self.anatom_site_general2id[anatom_site_general]
        
        value_list = []
        for key in self.label_dict.keys():
            # print(f'key is {key}: {skincon_row[key].item()}')
            value_list.append(skincon_row[key].item())
        total = sum(value_list)
        
        value_tensor = torch.tensor([x / total for x in value_list], dtype=torch.float32)
        
        final_label_id = self.label_dict[label.replace('_', ' ')]
        
        image_path = join(self.img_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        
        return img, value_tensor, torch.tensor(final_label_id, dtype=torch.int), sex_id, anatom_site_general_id, age
        
