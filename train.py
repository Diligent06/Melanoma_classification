from dataloader import MedicalDataset
from torch.utils.data import Dataset, DataLoader

dataset = MedicalDataset(csv_path='../datasets/SkinCON/data/train.csv')

train_loader = DataLoader(
    dataset,
    batch_size=2,   # 可根据 GPU 调整
    shuffle=True,
    num_workers=2
)


for batch_idx, (images, value_tensors, labels, sex_ids, anatom_site_general_ids, ages) in enumerate(train_loader):
    from IPython import embed; embed()

