import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_device():
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

DEVICE = get_device()

class FlattopDataset(Dataset):
    def __init__(self, csv_file, img_dir, target_size=(500, 500)):
        self.img_dir = img_dir
        self.target_size = target_size
        self.data = []
        
        # Read CSV
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader) # Skip header
            for row in reader:
                if not row: continue
                filename = row[0]
                # Read Z2 to Z15 (skip Z1 which is Piston)
                coeffs = [float(x) for x in row[2:]]
                self.data.append((filename, coeffs))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        filename, coeffs = self.data[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # Open image and convert to grayscale
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor: [1, H, W]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        # Resize to target size using interpolate
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32)
        return img_tensor, coeffs_tensor

class ZernikeResNet(nn.Module):
    def __init__(self, num_predict_modes=14):
        super().__init__()
        import torchvision.models as models
        self.resnet = models.resnet34(weights=None)
        # Modify first layer to accept 1 channel grayscale instead of 3 channel RGB
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final FC layer for 14 Zernike modes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_predict_modes)
        
    def forward(self, x): 
        return self.resnet(x)

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_mae, total_n = 0.0, 0.0, 0
    
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        mae = (pred - y).abs().mean().item()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += mae * bs
        total_n += bs
        
    return total_loss / total_n, total_mae / total_n

def main():
    dataset_dir = '../data/dataset'
    csv_file = os.path.join(dataset_dir, 'labels.csv')
    outdir = '../outputs/models'
    img_outdir = '../outputs/images'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(img_outdir, exist_ok=True)
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run dataset_generator.py first to generate data.")
        return
        
    print("Loading dataset...")
    dataset = FlattopDataset(csv_file, dataset_dir, target_size=(500, 500))
    print(f"Total samples found: {len(dataset)}")
    
    if len(dataset) < 100:
        print("WARNING: You have very few samples! The model will overfit. Please generate at least 10,000 samples for real training.")
        
    # 80/20 train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    
    # Check if dataset is too small to split
    if n_train == 0 or n_val == 0:
        print("Dataset too small to split into train/val. Use the entire dataset for both for testing purposes.")
        train_set = dataset
        val_set = dataset
    else:
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
        
    # Batch size: if dataset is tiny, use smaller batch size
    batch_size = min(16, max(1, len(dataset) // 4))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = ZernikeResNet(num_predict_modes=14).to(DEVICE)
    print(f"Training on device: {DEVICE}")
    
    criterion = nn.SmoothL1Loss()
    # 加入 L2 正則化 (weight_decay) 避免過擬合
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20  # 增加訓練週期讓模型充分收斂
    # 加入餘弦退火學習率排程 (Cosine Annealing LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val = float('inf')
    best_state = None
    
    for ep in range(1, epochs + 1):
        t_loss, t_mae = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_mae = run_epoch(model, val_loader, criterion, optimizer=None)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_mae'].append(t_mae)
        history['val_mae'].append(v_mae)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep:02d}/{epochs} | LR: {current_lr:.1e} | train loss: {t_loss:.4f}, val loss: {v_loss:.4f} | train mae: {t_mae:.4f}, val mae: {v_mae:.4f}")
        
        # 更新學習率
        scheduler.step()
        
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(outdir, 'resnet_stage1.pth'))
        print(f"\nBest model saved to {outdir}/resnet_stage1.pth (Val Loss: {best_val:.4f})")
        
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss (Smooth L1)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train')
    plt.plot(history['val_mae'], label='Val')
    plt.title('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_outdir, 'training_curve.png'), dpi=150)
    print(f"Training curves saved to {img_outdir}/training_curve.png")

if __name__ == '__main__':
    main()
