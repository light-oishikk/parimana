"""
Run full experiments: 3 models x 4 sampling ratios.
Saves results to results/experiment_results.json and prints a summary table.
"""
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_imaging.dataset import GhostImagingDataset
from src.models.fcn import FCN
from src.models.cnn import GICNN
from src.models.unet import GIUNet
from src.utils import calculate_metrics

IMAGE_SIZE = 28
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 784
SAMPLING_RATIOS = [0.0625, 0.125, 0.25, 0.50]  # 6.25%, 12.5%, 25%, 50%
MODELS = ['fcn', 'cnn', 'unet']
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3

def get_model(name, n_measurements):
    if name == 'fcn':
        return FCN(n_measurements=n_measurements, image_size=IMAGE_SIZE)
    elif name == 'cnn':
        return GICNN(n_measurements=n_measurements, image_size=IMAGE_SIZE)
    elif name == 'unet':
        return GIUNet(n_measurements=n_measurements, image_size=IMAGE_SIZE)

def train_and_eval(model_name, n_measurements, device):
    train_dataset = GhostImagingDataset(root='./data', n_measurements=n_measurements, image_size=IMAGE_SIZE, train=True)
    val_dataset = GhostImagingDataset(root='./data', n_measurements=n_measurements, image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model(model_name, n_measurements).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for measurements, targets in train_loader:
            measurements, targets = measurements.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(measurements)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        n_batches = 0
        with torch.no_grad():
            for measurements, targets in val_loader:
                measurements, targets = measurements.to(device), targets.to(device)
                outputs = model(measurements)
                psnr, ssim = calculate_metrics(outputs, targets)
                total_psnr += psnr
                total_ssim += ssim
                n_batches += 1

        avg_psnr = total_psnr / n_batches
        avg_ssim = total_ssim / n_batches

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            # Save best model
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'./checkpoints/{model_name}_M{n_measurements}_best.pth')

        print(f"  Epoch {epoch+1}/{EPOCHS} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    return best_psnr, best_ssim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Models: {MODELS}")
    print(f"Sampling ratios: {SAMPLING_RATIOS}")
    print(f"Epochs per run: {EPOCHS}")
    print("=" * 60)

    results = {}

    for ratio in SAMPLING_RATIOS:
        n_measurements = int(TOTAL_PIXELS * ratio)
        print(f"\n--- Sampling ratio: {ratio*100:.1f}% (M={n_measurements}) ---")

        for model_name in MODELS:
            print(f"\nTraining {model_name.upper()} with M={n_measurements}...")
            start = time.time()
            psnr, ssim = train_and_eval(model_name, n_measurements, device)
            elapsed = time.time() - start

            key = f"{model_name}_ratio{ratio}"
            results[key] = {
                'model': model_name,
                'sampling_ratio': ratio,
                'n_measurements': n_measurements,
                'best_psnr': round(psnr, 2),
                'best_ssim': round(ssim, 4),
                'training_time_s': round(elapsed, 1)
            }
            print(f"  DONE: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Time={elapsed:.0f}s")

    # Save results
    os.makedirs('./results', exist_ok=True)
    with open('./results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<8} {'Ratio':<8} {'M':<6} {'PSNR (dB)':<12} {'SSIM':<10} {'Time (s)':<10}")
    print("-" * 70)
    for r in results.values():
        print(f"{r['model'].upper():<8} {r['sampling_ratio']*100:.1f}%{'':<4} {r['n_measurements']:<6} {r['best_psnr']:<12} {r['best_ssim']:<10} {r['training_time_s']:<10}")
    print("=" * 70)

if __name__ == '__main__':
    main()
