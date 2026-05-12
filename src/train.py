import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ghost_imaging.dataset import GhostImagingDataset, PrecomputedGIDataset
from src.models.fcn import FCN
from src.models.cnn import GICNN
from src.models.unet import GIUNet
from src.utils import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Ghost Imaging Reconstruction Model")
    parser.add_argument('--model', type=str, default='fcn', choices=['fcn', 'cnn', 'unet'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--measurements', type=int, default=98, help='Number of measurements (M)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--precompute', action='store_true', help='Precompute dataset for speed')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    print(f"Loading dataset (M={args.measurements}, Size={args.image_size}x{args.image_size})...")
    if args.precompute:
        train_dataset = PrecomputedGIDataset(root=args.data_dir, n_measurements=args.measurements, image_size=args.image_size, train=True)
        val_dataset = PrecomputedGIDataset(root=args.data_dir, n_measurements=args.measurements, image_size=args.image_size, train=False)
    else:
        train_dataset = GhostImagingDataset(root=args.data_dir, n_measurements=args.measurements, image_size=args.image_size, train=True)
        val_dataset = GhostImagingDataset(root=args.data_dir, n_measurements=args.measurements, image_size=args.image_size, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    if args.model == 'fcn':
        model = FCN(n_measurements=args.measurements, image_size=args.image_size).to(device)
    elif args.model == 'cnn':
        model = GICNN(n_measurements=args.measurements, image_size=args.image_size).to(device)
    elif args.model == 'unet':
        model = GIUNet(n_measurements=args.measurements, image_size=args.image_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (measurements, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            measurements, targets = measurements.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(measurements)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for measurements, targets in val_loader:
                measurements, targets = measurements.to(device), targets.to(device)
                outputs = model(measurements)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                psnr, _ = calculate_metrics(outputs, targets)
                val_psnr += psnr
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val PSNR: {avg_val_psnr:.2f} dB")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == '__main__':
    main()
