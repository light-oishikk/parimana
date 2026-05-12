import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    Calculate PSNR and SSIM for a batch of images.
    Args:
        pred: (B, 1, H, W) tensor, values in [0, 1]
        target: (B, 1, H, W) tensor, values in [0, 1]
    Returns:
        avg_psnr, avg_ssim
    """
    pred_np = pred.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()
    
    if pred_np.ndim == 2:
        pred_np = np.expand_dims(pred_np, axis=0)
        target_np = np.expand_dims(target_np, axis=0)
        
    psnr_val = 0.0
    ssim_val = 0.0
    batch_size = pred_np.shape[0]
    
    for i in range(batch_size):
        p = pred_np[i]
        t = target_np[i]
        psnr_val += psnr(t, p, data_range=1.0)
        ssim_val += ssim(t, p, data_range=1.0)
        
    return psnr_val / batch_size, ssim_val / batch_size

def plot_reconstructions(target, pred, traditional=None, save_path=None):
    """
    Plot comparison of ground truth, DL prediction, and (optional) traditional reconstruction.
    """
    target = target.detach().cpu().numpy().squeeze()
    pred = pred.detach().cpu().numpy().squeeze()
    
    n_cols = 3 if traditional is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    if n_cols == 2:
        ax_gt, ax_pred = axes
    else:
        ax_gt, ax_pred, ax_trad = axes
        traditional = traditional.squeeze()
        ax_trad.imshow(traditional, cmap='gray')
        ax_trad.set_title('Traditional (Correlation)')
        ax_trad.axis('off')
        
    ax_gt.imshow(target, cmap='gray')
    ax_gt.set_title('Ground Truth')
    ax_gt.axis('off')
    
    ax_pred.imshow(pred, cmap='gray')
    ax_pred.set_title('DL Reconstruction')
    ax_pred.axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
