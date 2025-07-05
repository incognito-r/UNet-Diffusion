import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from utils.ema import create_ema_model
from utils.checkpoint import save_training_state, load_training_state
from utils.celeba_with_caption import CelebAloader
from utils.metrics.gpu import init_nvml, gpu_info
from utils.loss.lpips import safe_lpips
from omegaconf import OmegaConf
from utils.generate_samples import generate_sample

def main():
    torch.manual_seed(1)
    handle = init_nvml()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Enable mixed precision training
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    print("Mixed precision training enabled" if device == "cuda" else "Mixed precision training disabled")

    # Load configuration
    config = OmegaConf.load("configs/train_config_256.yaml")
    # config = OmegaConf.load("configs/temp.yaml")
    print(f"Configuration loaded: {OmegaConf.to_yaml(config)}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

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
    unet_ema_model, ema = create_ema_model(model, beta=config.training.ema_beta, step_start_ema=config.training.step_start_ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    MSE_LOSS = torch.nn.MSELoss()
    use_lpips = config.training.use_lpips
    if use_lpips:
        import lpips
        LPIPS_LOSS   = lpips.LPIPS(net=config.losses.lpips).to(device).eval() # net=vgg or alex

    print("Models, optimizers, losses initialized successfully.")
    #===================================================================

    # Dataset
    dataloader, _ = CelebAloader(data_config=config.data, train_config=config.training, device=device)
    print(f"Total Images: {len(dataloader.dataset)}, batch size: {dataloader.batch_size}")
    batch = next(iter(dataloader))
    print(f"Batch images shape: {batch['image'].shape}, Batch captions: {len(batch['text'])}, Batch images path: {len(batch['image_id'])}")

    # Checkpoint
    os.makedirs(config.checkpoint.path, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint.path, config.checkpoint.ckpt_name)
    start_epoch, best_loss = load_training_state(ckpt_path, model, optimizer, device)

    # === Training loop ===
    warmup_ep = config.training.warmup_epochs
    for epoch in range(start_epoch, config.training.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.train()

        cumm_loss = 0.0
        cumm_mse = 0.0
        cumm_lpips = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{config.training.epochs}")

        for batch_idx, batch in pbar:
            if batch_idx % config.training.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            images = batch['image'].to(device).float()
            captions = batch['text']
            text_inputs = clip_tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)

            with torch.no_grad():
                # ---- VAE Encoding ----
                latents = vae.encode(images).latent_dist.sample() * 0.18215 
                text_embeddings = clip_encoder(**text_inputs).last_hidden_state

            t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
            noise = torch.randn_like(latents)
            x_t = scheduler.add_noise(latents, noise, t)

            # ---- Noise Prediction ----
            with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
                noise_pred = model(x_t, timestep=t, encoder_hidden_states=text_embeddings).sample
                mse_loss = MSE_LOSS(noise_pred, noise) / config.training.grad_accum_steps

                if use_lpips:
                    # lpips weight
                    if epoch <= warmup_ep:
                        lpips_weight = 0.0
                    else:
                        # ramp-to-0.05 over [warmup_ep+1 .. 30], then hold
                        frac = min((epoch - warmup_ep) / float(30 - warmup_ep), 1.0)
                        lpips_weight = 0.05 * frac
                    # lpips loss
                    if lpips_weight > 0:
                        alpha_t = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).clamp(min=1e-7)
                        pred_x0 = (x_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                        normed_x0 = torch.nan_to_num(pred_x0 / 0.18215) # root cause
                        # normed_x0 = normed_x0.clamp(-6, 6)  # This solves the issue if nan in lpips but need optimum range for normed_x0 if any issue

                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda', enabled=False):
                                pred_rgb = vae.decode(normed_x0).sample.clamp(-1, 1)

                        if torch.isnan(pred_rgb).any() or torch.isinf(pred_rgb).any():
                            print(f"[FATAL] pred_rgb exploded at epoch {epoch}, step {batch_idx+1}")
                            print(f"normed_x0 stats: min={normed_x0.min():.2f}, max={normed_x0.max():.2f}, std={normed_x0.std():.2f}")
                            raise ValueError("pred_rgb contains NaNs or Infs")

                        lpips_loss =  LPIPS_LOSS(pred_rgb, images).mean()
                    else:
                        lpips_loss = torch.tensor(0.0, device=device)

                    total_loss = mse_loss + lpips_weight * lpips_loss
                
                else:
                    total_loss = mse_loss

            # ---- Backward Pass ----
            scaler.scale(total_loss).backward() # Overall loss

            # ---- Gradient Accumulation ----
            if (batch_idx + 1) % config.training.grad_accum_steps == 0:
                if device == 'cuda':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.step_ema(unet_ema_model, model)
                    
            # ---- Progress Tracking ----
            if use_lpips:
                cumm_lpips += lpips_loss.item() if isinstance(lpips_loss, float) else lpips_loss.item()
                avg_lpips = cumm_lpips / (batch_idx + 1)
            
            cumm_mse += mse_loss.item()
            avg_mse = cumm_mse / (batch_idx + 1)
            cumm_loss += total_loss.item() # main
            avg_loss = cumm_loss / (batch_idx + 1) # Total average loss
            best_loss = min(best_loss, avg_loss)
            avg_lpips = avg_lpips if use_lpips else 0.0

            pbar.set_postfix(avg_loss = avg_loss, avg_lpips = avg_lpips, GPU=gpu_info(handle))
        
            if (batch_idx+1) % 1 == 0:
                # normed_x0 min/max: {normed_x0.min().item():.3f}/{normed_x0.max().item():.3f} # Add logging for debugging
                logging.info(f"Epoch: {epoch} Batch: {batch_idx+1} | AVG MSE: {avg_mse:.4f} | AVG lpips: {avg_lpips:.4f} | AVG Total: {avg_loss:.4f}")

         # Epoch summary logging
        avg_loss = cumm_loss / (batch_idx + 1)
        print(f"Epoch {epoch} done. Avg loss: {avg_loss:.4f}")

        print("SAVING MODEL STATES...")
        # Save checkpoint & EMA weights
        save_training_state(
            checkpoint_path=ckpt_path, epoch=epoch,
            model=model, optimizer=optimizer,
            avg_loss=avg_loss, best_loss=best_loss
        )
        print("UNET MODEL SAVED!")

        ema_path = os.path.join(config.checkpoint.path, config.checkpoint.ema_ckpt_name)
        torch.save(unet_ema_model.state_dict(), ema_path)
        print("EMA_UNET MODEL SAVED!")

        # Generate visual sample for this epoch
        generate_sample(
            epoch=epoch, vae=vae, ema_model=unet_ema_model,
            scheduler=scheduler, tokenizer=clip_tokenizer, text_encoder=clip_encoder,
            config=config, device=device
        )
    
    print("Training completed.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/training.log")
        ]
    )
    main()
