
import torch

def safe_lpips(pred_rgb, target_rgb, lpips_model, device):
    """
    Computes LPIPS loss safely, avoiding NaNs or Infs.
    """
    def is_uniform(x):
        return (x == x[:, :, 0:1, 0:1]).all()

    pred_rgb = pred_rgb.detach().clamp(-1, 1)
    target_rgb = target_rgb.detach().clamp(-1, 1)

    if (
        torch.isnan(pred_rgb).any() or torch.isinf(pred_rgb).any() or
        torch.isnan(target_rgb).any() or torch.isinf(target_rgb).any()
    ):
        print("⚠️ Invalid values in LPIPS input. Skipping batch.")
        return torch.tensor(0.0, device=device)

    if is_uniform(pred_rgb) or is_uniform(target_rgb):
        print("⚠️ One of the LPIPS inputs is uniform. Skipping batch.")
        return torch.tensor(0.0, device=device)

    with torch.no_grad():
        val = lpips_model(pred_rgb, target_rgb).mean()

    if torch.isnan(val) or torch.isinf(val):
        print("⚠️ LPIPS returned NaN/Inf. Skipping batch.")
        return torch.tensor(0.0, device=device)

    return val


