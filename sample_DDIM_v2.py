import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from omegaconf import OmegaConf
from utils.ema import create_ema_model
from utils.metrics.gpu import init_nvml, gpu_info

@torch.no_grad()
def main():
    # Load configuration
    config = OmegaConf.load("configs/train_config_256.yaml")
    model_cfg = config.model
    sample_cfg = config.sampling

    device = "cuda" if torch.cuda.is_available() else "cpu"
    handle = init_nvml()

    # Prepare output
    os.makedirs(sample_cfg.dir, exist_ok=True)
    output_path = os.path.join(sample_cfg.dir, 'test_sample_grid.png')

    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    # Load conditional U-Net
    model = UNet2DConditionModel(
        sample_size=model_cfg.sample_size,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        down_block_types=tuple(model_cfg.down_block_types),
        up_block_types=tuple(model_cfg.up_block_types),
        block_out_channels=tuple(model_cfg.block_out_channels),
        layers_per_block=model_cfg.layers_per_block,
        cross_attention_dim=model_cfg.cross_attention_dim,
    ).to(device)

    # EMA wrapper
    ema_model, _ = create_ema_model(
        model,
        beta=config.training.ema_beta,
        step_start_ema=config.training.step_start_ema
    )
    ema_ckpt = os.path.join(config.checkpoint.path, config.checkpoint.ema_ckpt_name)
    ema_model.load_state_dict(torch.load(ema_ckpt, map_location=device))
    ema_model.eval()

    # Scheduler aligned with training
    scheduler = DDPMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule=config.scheduler.type
    )
    scheduler.set_timesteps(sample_cfg.steps)

    # Load CLIP
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()

    # Prepare guidance embeddings
    prompt = "a beautiful woman in a red dress"
    text_inputs = tokenizer(
        [prompt] * sample_cfg.num_samples,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    text_emb = text_encoder(**text_inputs).last_hidden_state
    # Unconditional (empty) embeddings
    uncond_inputs = tokenizer(
        [""] * sample_cfg.num_samples,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    uncond_emb = text_encoder(**uncond_inputs).last_hidden_state

    # Sampling
    num_samples = sample_cfg.num_samples
    shape = (num_samples, model_cfg.in_channels, model_cfg.sample_size, model_cfg.sample_size)
    latents = torch.randn(shape, device=device)
    guidance_scale = sample_cfg.guidance_scale if hasattr(sample_cfg, 'guidance_scale') else 7.5

    for t in tqdm(scheduler.timesteps, desc="Sampling"):  # timesteps descends
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise for both conditional and unconditional
        eps_uncond = ema_model(latents, timestep=t_batch, encoder_hidden_states=uncond_emb).sample
        eps_cond   = ema_model(latents, timestep=t_batch, encoder_hidden_states=text_emb).sample
        # Classifier-free guidance
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Step
        latents = scheduler.step(eps, t, latents).prev_sample

    # Decode latents to images
    images = vae.decode(latents / 0.18215).sample
    images = (images.clamp(-1, 1) + 1) / 2

    # Save grid
    grid = make_grid(images, nrow=int(num_samples**0.5))
    save_image(grid, output_path)
    print(f"✅ Samples saved to {output_path}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
