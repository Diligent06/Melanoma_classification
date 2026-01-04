from dataloader import MedicalDataset
from torch.utils.data import Dataset, DataLoader
import torch
from model.model import Backbone_MoE
import torch.nn as nn
from tqdm import tqdm
import os
import wandb

# wandb.init(
#     project="Medical_MoE",
#     name="MoE_Training_Run",
#     mode="offline"
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

dataset = MedicalDataset(csv_path='../datasets/SkinCON/data/train.csv')
val_dataset = MedicalDataset(csv_path='../datasets/SkinCON/data/val.csv')

train_loader = DataLoader(
    dataset,
    batch_size=32,   # 可根据 GPU 调整
    shuffle=True,
    num_workers=8
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

model = Backbone_MoE(expert_num=4, device=device, dtype=dtype)
model._init_weights()
model.to(device=device, dtype=dtype)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
criterion_prob = nn.KLDivLoss(reduction='batchmean')

epoch_num = 1000
minimal_val_loss = float('inf')
save_folder = './checkpoints/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
for epoch in range(1, epoch_num + 1):
    model.train()
    model.training = True
    train_loss = 0.0
    correct = 0
    total = 0
    train_loss_out = 0.0
    train_loss_prob = 0.0
    train_loss_gate = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_num} - Training"):
        images = batch[0].to(device, dtype=dtype)
        value_tensor = batch[1].to(device, dtype=dtype)
        final_label_id = batch[2].to(device, dtype=torch.long)
        sex_id = batch[3].to(device, dtype=dtype)
        anatom_site_general_id = batch[4].to(device, dtype=dtype)
        age = batch[5].to(device, dtype=dtype)

        optimizer.zero_grad()
        context_features = {'age': age, 'sex': sex_id, 'anatom_site_general': anatom_site_general_id}
        prob, out, gate_w = model(images, context_features)
        loss_out = criterion(out, final_label_id)
        log_prob = torch.log_softmax(prob, dim=1)
            
        value_tensor = value_tensor / value_tensor.sum(dim=1, keepdim=True)
        value_tensor = value_tensor.clamp(min=1e-6)
        
        loss_prob = criterion_prob(log_prob, value_tensor)
        # loss_prob = criterion_prob(torch.log(prob + 1e-8), value_tensor)
        loss_gate = torch.sum(gate_w ** 2)
        loss = loss_out + 0.8 * loss_prob + loss_gate * 0.01
        
        loss.backward()
        optimizer.step()
        
        pred_index = torch.argmax(out, dim=1)
        correct += (pred_index == final_label_id).sum().item()
        train_loss += loss.item() * images.shape[0]
        train_loss_out += loss_out.item() * images.shape[0]
        train_loss_prob += loss_prob.item() * images.shape[0]
        train_loss_gate += loss_gate.item() * images.shape[0]
        total += images.shape[0]  # sample number
        
        # from IPython import embed; embed()
    
    train_loss /= total
    train_loss_out /= total
    train_loss_prob /= total
    train_loss_gate /= total
    train_acc = correct / total
    
    if epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join(save_folder, f'model_epoch_{epoch}.pth'))
        
    model.eval()
    model.training = False
    val_loss = 0.0
    val_loss_out = 0.0
    val_loss_prob = 0.0
    val_loss_gate = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epoch_num} - Validation"):
            images = batch[0].to(device, dtype=dtype)
            value_tensor = batch[1].to(device, dtype=dtype)
            final_label_id = batch[2].to(device, dtype=torch.long)
            sex_id = batch[3].to(device, dtype=dtype)
            anatom_site_general_id = batch[4].to(device, dtype=dtype)
            age = batch[5].to(device, dtype=dtype)

            context_features = {'age': age, 'sex': sex_id, 'anatom_site_general': anatom_site_general_id}
            prob, out, gate_w = model(images, context_features)
            val_loss_out = criterion(out, final_label_id)
            log_prob = torch.log_softmax(prob, dim=1)
            
            value_tensor = value_tensor / value_tensor.sum(dim=1, keepdim=True)
            value_tensor = value_tensor.clamp(min=1e-6)
            
            val_loss_prob = criterion_prob(log_prob, value_tensor)
            val_loss_gate = torch.sum(gate_w ** 2)
            val_loss = val_loss_out + 0.8 * val_loss_prob + val_loss_gate * 0.01
            
            pred_index = torch.argmax(out, dim=1)
            correct += (pred_index == final_label_id).sum().item()
            val_loss += val_loss.item() * images.shape[0]
            val_loss_out += val_loss_out.item() * images.shape[0]
            val_loss_prob += val_loss_prob.item() * images.shape[0]
            val_loss_gate += val_loss_gate.item() * images.shape[0]
            total += images.shape[0]  # sample number
            
            
        
    val_loss /= total
    val_loss_out /= total
    val_loss_prob /= total
    val_loss_gate /= total
    val_acc = correct / total
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_loss_out": train_loss_out,
        "train_loss_prob": train_loss_prob,
        "train_loss_gate": train_loss_gate,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_loss_out": val_loss_out,
        "val_loss_prob": val_loss_prob,
        "val_loss_gate": val_loss_gate
    })
    
    if val_loss < minimal_val_loss:
        minimal_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(save_folder, f'best_model_epoch.pth'))
    
    print(f"Epoch [{epoch}/{epoch_num}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
          f"Train Loss Out: {train_loss_out:.4f}, Train Loss Prob: {train_loss_prob:.4f}, Train Loss Gate: {train_loss_gate:.4f}"
          f"Val Loss Out: {val_loss_out:.4f}, Val Loss Prob: {val_loss_prob:.4f}, Val Loss Gate: {val_loss_gate:.4f}")
    





