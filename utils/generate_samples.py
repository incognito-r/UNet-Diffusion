import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

def generate_sample(epoch: int, vae, ema_model, scheduler, tokenizer, text_encoder, config, device):
    ema_model.eval()
    text_encoder.eval()

    prompt = "a beautiful woman in a red dress"
    batch_size = 4

    # Tokenize prompt
    text_inputs = tokenizer([prompt] * batch_size, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(**text_inputs).last_hidden_state

    # Prepare noise
    latents = torch.randn((batch_size, config.model.in_channels, config.model.sample_size, config.model.sample_size)).to(device)

    # Denoising loop
    scheduler.set_timesteps(50)
    for t in tqdm(scheduler.timesteps, desc=f"Sampling epoch {epoch}"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
            noise_pred = ema_model(latents, timestep=t_batch, encoder_hidden_states=text_embeddings).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode and save
    imgs = vae.decode(latents / 0.18215).sample.clamp(-1, 1)
    imgs = (imgs + 1) / 2
    grid = make_grid(imgs, nrow=2)
    sample_dir = os.path.join(config.checkpoint.path, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    save_image(grid, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"))
    print(f"✅ Saved sample grid for epoch {epoch}")
