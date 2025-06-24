
import torch

def safe_lpips(pred_rgb, target_rgb, lpips_model, device):
    """
    Computes LPIPS loss safely, avoiding NaNs or Infs.

    Args:
        pred_rgb (torch.Tensor): Predicted images in [-1, 1], shape [B, 3, H, W]
        target_rgb (torch.Tensor): Ground-truth images in [-1, 1], shape [B, 3, H, W]
        lpips_model (lpips.LPIPS): An instance of the LPIPS model
        device (str or torch.device): Device for returning zero loss if invalid

    Returns:
        torch.Tensor: Scalar LPIPS loss or 0.0 if NaN/Inf
    """
    with torch.no_grad():
        val = lpips_model(pred_rgb, target_rgb).mean()

    if torch.isnan(val) or torch.isinf(val):
        print("⚠️ LPIPS loss returned NaN or Inf. Skipping this batch.")
        return torch.tensor(0.0, device=device)
    return val

