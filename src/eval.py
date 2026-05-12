import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.ghost_imaging.dataset import GhostImagingDataset
from src.models.fcn import FCN
from src.models.cnn import GICNN
from src.models.unet import GIUNet
from src.utils import calculate_metrics, plot_reconstructions

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ghost Imaging Reconstruction Model")
    parser.add_argument('--model', type=str, default='fcn', choices=['fcn', 'cnn', 'unet'])
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--measurements', type=int, default=98)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = GhostImagingDataset(root=args.data_dir, n_measurements=args.measurements, image_size=args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load model
    if args.model == 'fcn':
        model = FCN(n_measurements=args.measurements, image_size=args.image_size).to(device)
    elif args.model == 'cnn':
        model = GICNN(n_measurements=args.measurements, image_size=args.image_size).to(device)
    elif args.model == 'unet':
        model = GIUNet(n_measurements=args.measurements, image_size=args.image_size).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    
    print("Evaluating...")
    with torch.no_grad():
        for i, (measurements, targets) in enumerate(loader):
            measurements, targets = measurements.to(device), targets.to(device)
            outputs = model(measurements)
            
            psnr, ssim = calculate_metrics(outputs, targets)
            total_psnr += psnr
            total_ssim += ssim
            
            # Save some examples from the first batch
            if i == 0:
                for j in range(min(5, outputs.size(0))):
                    # Also compute traditional reconstruction for comparison
                    bucket_np = measurements[j].cpu().numpy()
                    target_np = targets[j].cpu().numpy().squeeze()
                    
                    # Need patterns for traditional recon
                    patterns = dataset.patterns if dataset.precompute_patterns else dataset.simulator.generate_patterns(args.measurements)
                    trad_recon = dataset.simulator.traditional_reconstruct(patterns, bucket_np)
                    
                    plot_reconstructions(
                        targets[j], 
                        outputs[j], 
                        traditional=trad_recon,
                        save_path=os.path.join(args.save_dir, f'sample_{j}.png')
                    )

    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    
    print(f"Results for {args.model} with {args.measurements} measurements:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Measurements: {args.measurements}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

if __name__ == '__main__':
    main()
