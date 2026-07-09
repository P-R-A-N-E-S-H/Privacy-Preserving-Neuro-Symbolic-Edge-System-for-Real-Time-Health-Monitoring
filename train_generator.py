import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from diffusion_generator import DiffusionDenoiser, TimeSeriesDDPM
from sepsis_logic import SepsisKnowledgeBase, logical_loss
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# STABILITY HELPER FUNCTIONS
# ==========================================
def clean_data(X, y):
    """Remove any samples containing NaN or Inf"""
    nan_mask = torch.isnan(X).reshape(X.shape[0], -1).any(dim=1)
    inf_mask = torch.isinf(X).reshape(X.shape[0], -1).any(dim=1)
    bad_mask = nan_mask | inf_mask
    
    if bad_mask.any():
        print(f"⚠ Found {bad_mask.sum().item()} corrupted samples in training data. Removing them...")
        return X[~bad_mask], y[~bad_mask]
    return X, y

def train_stable_generator(
    train_data_path='combined_train.pt',
    n_epochs=30,      # Reduced epochs since dataset is huge (61k)
    batch_size=64,    # Increased batch size for stability
    lr=2e-5,          # REDUCED LR to prevent explosion
    lambda_ce=1.0,
    lambda_logic=0.1, # REDUCED logic weight to prevent fighting
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print("="*80)
    print("STARTING STABLE GENERATOR TRAINING")
    print("="*80)

    # 1. LOAD DATA
    print(f"Loading data from {train_data_path}...")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Cannot find {train_data_path}. Did you run combine_data.py?")
        
    data = torch.load(train_data_path, weights_only=True)
    X = data['X']
    y = data['y']
    
    # 2. CLEAN DATA IMMEDIATELY
    X, y = clean_data(X, y)
    
    print(f"Data loaded: {len(X)} samples")
    
    # 3. CLIP EXTREME OUTLIERS (Physiological Safety)
    # HR: 20-200, BP: 40-250, SpO2: 50-100, Shock: 0-5
    # We clip loosely to 0-300 to remove artifacts like 99999
    X = torch.clamp(X, min=0.0, max=300.0) 

    # 4. NORMALIZE
    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True) + 1e-4
    X_normalized = (X - X_mean) / X_std
    
    # Create dataset
    dataset = TensorDataset(X_normalized, y)
    # drop_last=True is crucial for stability with BatchNorm/GroupNorm
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 

    # 5. INITIALIZE MODELS
    print("Initializing models...")
    denoiser = DiffusionDenoiser(seq_len=24, n_features=5, hidden_dim=128).to(device)
    ddpm = TimeSeriesDDPM(denoiser, n_timesteps=1000, device=device)
    knowledge_base = SepsisKnowledgeBase()

    # Weight decay helps stabilize weights
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-5)
    
    # OneCycleLR is safer than constant LR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr*10, 
        steps_per_epoch=len(dataloader), 
        epochs=n_epochs
    )

    # 6. TRAINING LOOP
    print(f"\nTraining for {n_epochs} epochs on {device}...")
    
    for epoch in range(n_epochs):
        denoiser.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()

            optimizer.zero_grad()
            
            # --- Forward Pass ---
            t = torch.randint(0, ddpm.n_timesteps, (batch_x.shape[0],)).to(device)
            x_noisy, true_noise = ddpm.add_noise(batch_x, t)
            predicted_noise = denoiser(x_noisy, t, batch_y)
            
            # --- Diffusion Loss ---
            diff_loss = nn.MSELoss()(predicted_noise, true_noise)
            
            # --- Logic Loss (Safe Estimate) ---
            # We estimate the clean signal to check physiological validity
            # x_clean ≈ x_noisy - predicted_noise (Simplified for stability)
            x_pred_clean = x_noisy - predicted_noise 
            x_pred_denorm = x_pred_clean * X_std.to(device) + X_mean.to(device)
            
            # Calculate logic consistency
            log_loss = logical_loss(x_pred_denorm, batch_y.float(), knowledge_base)
            
            # Combine losses
            total_loss = lambda_ce * diff_loss + lambda_logic * log_loss

            # --- SAFETY CHECK ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                # Just skip this batch, don't crash
                optimizer.zero_grad()
                continue
            
            total_loss.backward()
            
            # --- GRADIENT CLIPPING (The Anti-Explosion Shield) ---
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}', 
                'logic': f'{log_loss.item():.4f}'
            })

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'denoiser_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std
            }, f'generator_checkpoint_epoch_{epoch+1}.pt')
            print(f"  Checkpoint saved!")

    print("\n" + "="*80)
    print("✓ STABLE TRAINING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    train_stable_generator()