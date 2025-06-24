import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from utils.ema import create_ema_model
from utils.checkpoint import save_training_state, load_training_state
from utils.celeba_with_caption import CelebAloader
from utils.metrics.gpu import init_nvml, gpu_info
from omegaconf import OmegaConf
import lpips
from sample_inference import generate_sample

def main():
    torch.manual_seed(1)
    handle = init_nvml()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    print("Mixed precision training enabled" if device == "cuda" else "Mixed precision training disabled")

    # Load configuration
    config = OmegaConf.load("configs/train_config_256.yaml")
    print(f"Configuration loaded: {OmegaConf.to_yaml(config)}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    # Load UNet
    model = UNet2DConditionModel(
        sample_size=config.model.sample_size,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        down_block_types=config.model.down_block_types,
        up_block_types=config.model.up_block_types,
        block_out_channels=config.model.block_out_channels,
        layers_per_block=config.model.layers_per_block,
        cross_attention_dim=config.model.cross_attention_dim,
    ).to(device)

    # Noise scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule=config.scheduler.type,
    )

    # CLIP tokenizer & encoder
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    # EMA
    unet_ema_model, ema = create_ema_model(
        model,
        beta=config.training.ema_beta,
        step_start_ema=config.training.step_start_ema
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    MSE_LOSS = torch.nn.MSELoss()
    LPIPS_LOSS = lpips.LPIPS(net='vgg').to(device).eval()

    # Data
    dataloader, _ = CelebAloader(data_config=config.data, train_config=config.training)
    print(f"Dataset size: {len(dataloader.dataset)} images, batch size: {dataloader.batch_size}")

    # Checkpoint
    os.makedirs(config.checkpoint.path, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint.path, config.checkpoint.ckpt_name)
    start_epoch, best_loss = load_training_state(ckpt_path, model, optimizer, device)
    print(f"Resuming from epoch {start_epoch}, best_loss {best_loss:.4f}")

    warmup_ep = config.training.warmup_epochs
    for epoch in range(start_epoch, config.training.epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config.training.epochs}")
        for batch_idx, batch in pbar:
            if batch_idx % config.training.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            images = batch['image'].to(device).float()
            captions = batch['caption']
            text_inputs = clip_tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                text_embeddings = clip_encoder(**text_inputs).last_hidden_state
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
            noise = torch.randn_like(latents)
            x_t = scheduler.add_noise(latents, noise, t)

            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                noise_pred = model(x_t, timestep=t, encoder_hidden_states=text_embeddings).sample
                mse_loss = MSE_LOSS(noise_pred, noise) / config.training.grad_accum_steps

                if epoch+1 <= warmup_ep:
                    lpips_weight = 0.0
                else:
                    # ramp-to-0.05 over [warmup_ep+1 .. 30], then hold
                    frac = min((epoch+1 - warmup_ep) / float(30 - warmup_ep), 1.0)
                    lpips_weight = 0.05 * frac

                if lpips_weight > 0:
                    pred_x0 = scheduler.step(noise_pred, t[0].item(), x_t).pred_original_sample
                    pred_rgb = vae.decode(pred_x0 / 0.18215).sample.clamp(-1, 1)
                    lpips_loss = LPIPS_LOSS(pred_rgb, images).mean()
                else:
                    lpips_loss = 0.0

                total_loss = mse_loss + lpips_weight * lpips_loss

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % config.training.grad_accum_steps == 0:
                if device == 'cuda':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.step_ema(unet_ema_model, model)

            # Log scaled loss per batch
            running_loss += total_loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            best_loss = min(best_loss, avg_loss)
            pbar.set_postfix(avg_loss=avg_loss, mem=gpu_info(handle))

        # Save checkpoint & EMA weights
        save_training_state(
            checkpoint_path=ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            avg_loss=avg_loss,
            best_loss=best_loss
        )
        ema_path = os.path.join(config.checkpoint.path, config.checkpoint.ema_ckpt_name)
        torch.save(unet_ema_model.state_dict(), ema_path)
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

        # Generate visual sample
        generate_sample(
            epoch=epoch+1,
            vae=vae,
            ema_model=unet_ema_model,
            scheduler=scheduler,
            tokenizer=clip_tokenizer,
            text_encoder=clip_encoder,
            config=config,
            device=device
        )

    print("Training completed.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    main()
