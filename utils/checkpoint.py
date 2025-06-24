import os
import torch
from datetime import datetime
import csv


def load_training_state(checkpoint_path, model, optimizer, discriminator=None, optimizer_d=None, 
                      lr_scheduler=None, device='cpu'):
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return 0, float('inf')
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"✅ Resuming from checkpoint: {checkpoint_path}")
    
    # Initialize best_loss
    best_loss = ckpt.get('best_loss', float('inf'))
    
    # Load Model Components
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print("Loaded base model and optimizer")

    # discriminator
    if discriminator and optimizer_d:
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
        print("Loaded discriminator and optimizer")

    # Handle scheduler
    if lr_scheduler:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        print("Loaded learning rate scheduler")
    
    start_epoch = ckpt["epoch"] + 1  # Start from NEXT epoch
    print(f"Resuming at epoch {start_epoch}, previous loss: {ckpt['loss']:.4f}")
    
    return start_epoch, best_loss

def save_training_state(checkpoint_path, epoch, model, optimizer, avg_loss, best_loss, 
                      discriminator=None, optimizer_d=None, lr_scheduler=None):
    """Save training state with discriminator support"""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss,
    }
    
    # Add discriminator if available
    if discriminator and optimizer_d:
        ckpt["discriminator_state_dict"] = discriminator.state_dict()
        ckpt["optimizer_d_state_dict"] = optimizer_d.state_dict()
        print("Discriminator Saved!!")
    
    # Add scheduler if available
    if lr_scheduler:
        ckpt["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        print("Learning Rate Scheduler Saved!!")

    # Save checkpoint
    torch.save(ckpt, checkpoint_path)
    