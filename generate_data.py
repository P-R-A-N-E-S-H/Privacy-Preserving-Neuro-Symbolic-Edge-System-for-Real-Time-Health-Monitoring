import torch
from diffusion_generator import DiffusionDenoiser, TimeSeriesDDPM
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_trained_generator(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load trained generator checkpoint
    """
    print(f"Loading generator from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    # Initialize model
    denoiser = DiffusionDenoiser(seq_len=24, n_features=5, hidden_dim=128).to(device)
    denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
    denoiser.eval()
    
    ddpm = TimeSeriesDDPM(denoiser, n_timesteps=1000, device=device)
    
    X_mean = checkpoint['X_mean'].to(device)
    X_std = checkpoint['X_std'].to(device)
    
    print(f"✓ Generator loaded (epoch {checkpoint['epoch']+1})")
    return ddpm, X_mean, X_std


def augment_single_client(client_path, ddpm, X_mean, X_std, target_ratio=0.3, device='cpu'):
    """
    Generate synthetic sepsis patients for one client
    """
    data = torch.load(client_path, weights_only=True)
    X_original = data['X']  # [n_samples, 24, 5]
    y_original = data['y']
    
    n_total = len(y_original)
    n_sepsis = (y_original == 1).sum().item()
    n_healthy = n_total - n_sepsis
    
    current_ratio = n_sepsis / n_total if n_total > 0 else 0
    
    # Check if augmentation needed
    if current_ratio >= target_ratio:
        return X_original, y_original, 0
    
    n_sepsis_needed = int(n_healthy * target_ratio / (1 - target_ratio)) - n_sepsis
    n_sepsis_needed = max(0, n_sepsis_needed)
    
    if n_sepsis_needed == 0:
        return X_original, y_original, 0
    
    print(f"  Generating {n_sepsis_needed} synthetic patients...")
    print(f"    Before: {n_sepsis}/{n_total} ({current_ratio:.1%}) sepsis")
    
    # Generate in batches
    batch_size = 32
    n_batches = (n_sepsis_needed + batch_size - 1) // batch_size
    
    synthetic_list = []
    
    for batch_idx in range(n_batches):
        batch_n = min(batch_size, n_sepsis_needed - batch_idx * batch_size)
        
        condition_labels = torch.ones(batch_n, dtype=torch.long).to(device)
        
        with torch.no_grad():
            synthetic_norm = ddpm.sample(
                condition_label=condition_labels,
                batch_size=batch_n,
                seq_len=24,
                n_features=5  # CRITICAL: Match your data
            )
            
            # Denormalize
            synthetic = synthetic_norm * X_std + X_mean
            
            # Clamp to physiological limits (5 features only)
            LIMITS = {
                0: (20, 200),    # HR
                1: (50, 250),    # SBP
                2: (30, 150),    # DBP
                3: (50, 100),    # SpO2
                4: (0.1, 3.0)    # Shock Index
            }
            
            for feat_idx, (min_val, max_val) in LIMITS.items():
                synthetic[:, :, feat_idx] = torch.clamp(
                    synthetic[:, :, feat_idx],
                    min_val,
                    max_val
                )
            
            synthetic_list.append(synthetic.cpu())
    
    # Combine
    synthetic_X = torch.cat(synthetic_list, dim=0)
    synthetic_y = torch.ones(n_sepsis_needed, dtype=torch.long)
    
    X_augmented = torch.cat([X_original, synthetic_X], dim=0)
    y_augmented = torch.cat([y_original, synthetic_y], dim=0)
    
    new_n_sepsis = (y_augmented == 1).sum().item()
    new_ratio = new_n_sepsis / len(y_augmented)
    
    print(f"    After: {new_n_sepsis}/{len(y_augmented)} ({new_ratio:.1%}) sepsis")
    
    return X_augmented, y_augmented, n_sepsis_needed


def augment_training_clients(
    client_dir,
    checkpoint_path,
    output_dir=None,
    target_ratio=0.3,
    max_client_id=79,  # CRITICAL: Only augment 0-79
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Augment ONLY training clients (0-79)
    Leave test clients (80-99) untouched
    """
    client_dir = Path(client_dir)
    
    if output_dir is None:
        output_dir = client_dir / "augmented"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Load generator
    ddpm, X_mean, X_std = load_trained_generator(checkpoint_path, device)
    
    # Find client files
    all_files = sorted(client_dir.glob("client_*.pt"))
    
    # FILTER: Only training clients (0 to max_client_id)
    train_files = []
    for f in all_files:
        client_id = int(f.stem.split('_')[1])
        if client_id <= max_client_id:
            train_files.append(f)
    
    if len(train_files) == 0:
        print(f"ERROR: No training clients found in {client_dir}")
        return
    
    print("="*80)
    print("DATA AUGMENTATION")
    print("="*80)
    print(f"\nAugmenting {len(train_files)} training clients (0-{max_client_id})")
    print(f"Target sepsis ratio: {target_ratio:.0%}")
    print(f"Output: {output_dir}\n")
    
    total_generated = 0
    
    for client_file in tqdm(train_files, desc="Augmenting"):
        X_aug, y_aug, n_gen = augment_single_client(
            client_file,
            ddpm,
            X_mean,
            X_std,
            target_ratio,
            device
        )
        
        total_generated += n_gen
        
        # Save
        output_path = output_dir / client_file.name
        torch.save({'X': X_aug, 'y': y_aug}, output_path)
    
    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)
    print(f"✓ Generated {total_generated:,} synthetic patients")
    print(f"✓ Saved to: {output_dir}")
    print(f"\n⚠ Test clients (80-99) remain UNTOUCHED in original directory")


if __name__ == "__main__":
    augment_training_clients(
        client_dir=r"D:\college\PROJECTS-SEM 4\mimic project\federated_clients",
        checkpoint_path=r"D:\college\PROJECTS-SEM 4\mimic project\generator_checkpoints_2\generator_checkpoint_epoch_30.pt",
        target_ratio=0.3,
        max_client_id=79  # Only augment clients 0-79
    )