import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_model import SepsisTransformer, count_parameters
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from pathlib import Path


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute comprehensive evaluation metrics
    """
    y_pred = (y_pred_probs[:, 1] > threshold).astype(int)
    
    auroc = roc_auc_score(y_true, y_pred_probs[:, 1])
    auprc = average_precision_score(y_true, y_pred_probs[:, 1])
    f1 = f1_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def load_federated_data(
    raw_dir,
    augmented_dir, 
    train_clients=80,
    total_clients=100
):
    """
    Load training data from augmented clients (0-79)
    Load test data from raw clients (80-99)
    """
    raw_dir = Path(raw_dir)
    augmented_dir = Path(augmented_dir)
    
    print("="*80)
    print("LOADING FEDERATED DATA")
    print("="*80)
    
    # TRAINING DATA: Augmented clients 0-79
    print(f"\nLoading TRAINING data from augmented clients (0-{train_clients-1})...")
    X_train_list = []
    y_train_list = []
    
    for i in tqdm(range(train_clients), desc="Train clients"):
        client_file = augmented_dir / f"client_{i}.pt"
        if not client_file.exists():
            continue
        
        data = torch.load(client_file, weights_only=True)
        X_train_list.append(data['X'])
        y_train_list.append(data['y'])
    
    if len(X_train_list) == 0:
        raise ValueError("No training data found! Check augmented_dir path.")
    
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    
    # TEST DATA: Raw clients 80-99
    print(f"Loading TEST data from raw clients ({train_clients}-{total_clients-1})...")
    X_test_list = []
    y_test_list = []
    
    for i in tqdm(range(train_clients, total_clients), desc="Test clients"):
        client_file = raw_dir / f"client_{i}.pt"
        if not client_file.exists():
            continue
        
        data = torch.load(client_file, weights_only=True)
        X_test_list.append(data['X'])
        y_test_list.append(data['y'])
    
    if len(X_test_list) == 0:
        raise ValueError("No test data found! Check raw_dir path.")
    
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    
    print("\n" + "="*80)
    print("DATA LOADING SUMMARY")
    print("="*80)
    print(f"Training samples: {len(X_train):,}")
    print(f"  Sepsis: {(y_train==1).sum().item():,} ({100*(y_train==1).sum().item()/len(y_train):.2f}%)")
    print(f"  Healthy: {(y_train==0).sum().item():,} ({100*(y_train==0).sum().item()/len(y_train):.2f}%)")
    
    print(f"\nTest samples: {len(X_test):,}")
    print(f"  Sepsis: {(y_test==1).sum().item():,} ({100*(y_test==1).sum().item()/len(y_test):.2f}%)")
    print(f"  Healthy: {(y_test==0).sum().item():,} ({100*(y_test==0).sum().item()/len(y_test):.2f}%)")
    print("="*80 + "\n")
    
    return X_train, y_train, X_test, y_test


def train_transformer(
    raw_dir,
    augmented_dir,
    n_epochs=30,
    batch_size=64,
    lr=5e-5,  # REDUCED from 1e-4
    weight_decay=1e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train Sepsis Transformer with numerical stability
    """
    
    print("="*80)
    print("SEPSIS PREDICTION MODEL TRAINING")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val = load_federated_data(raw_dir, augmented_dir)
    
    n_features = X_train.shape[2]
    print(f"\nFeature dimensions: {n_features}")
    
    if n_features != 5:
        raise ValueError(f"Expected 5 features, got {n_features}")
    
    # Normalize with extra stability
    print("\nNormalizing data...")
    X_mean = X_train.mean(dim=(0, 1), keepdim=True)
    X_std = X_train.std(dim=(0, 1), keepdim=True) + 1e-4
    
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    
    # CRITICAL: Clip extreme normalized values
    X_train_norm = torch.clamp(X_train_norm, -10, 10)
    X_val_norm = torch.clamp(X_val_norm, -10, 10)
    
    print(f"Normalized train stats: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
    print(f"Normalized train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_norm, y_train)
    val_dataset = TensorDataset(X_val_norm, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing Transformer...")
    model = SepsisTransformer(
        n_features=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.15  # INCREASED dropout for stability
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Class weights
    n_sepsis = (y_train == 1).sum().item()
    n_healthy = (y_train == 0).sum().item()
    class_weight = torch.tensor([1.0, n_healthy / n_sepsis]).to(device)
    
    print(f"Class weights: [1.00, {class_weight[1]:.2f}]")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )
    
    best_auprc = 0.0
    best_epoch = 0
    
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80 + "\n")
    
    for epoch in range(n_epochs):
        # TRAINING
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        n_nan_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            
            optimizer.zero_grad()
            
            logits = model(batch_x)
            
            # NaN check
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                n_nan_batches += 1
                continue
            
            loss = criterion(logits, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan_batches += 1
                continue
            
            loss.backward()
            
            # CRITICAL: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        if n_nan_batches > 0:
            print(f"\n  Warning: Skipped {n_nan_batches} NaN batches")
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / (len(train_loader) - n_nan_batches)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        all_y_true = []
        all_y_pred_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).long()
                
                logits = model(batch_x)
                
                # Skip NaN batches
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_y_true.extend(batch_y.cpu().numpy())
                all_y_pred_probs.append(probs)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if len(all_y_pred_probs) == 0:
            print("\n  ERROR: All validation batches had NaN!")
            break
        
        avg_val_loss = val_loss / len(val_loader)
        all_y_true = np.array(all_y_true)
        all_y_pred_probs = np.vstack(all_y_pred_probs)
        
        # Compute metrics
        metrics = compute_metrics(all_y_true, all_y_pred_probs)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{n_epochs} Summary")
        print(f"{'='*80}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"\nClinical Metrics:")
        print(f"  AUROC:       {metrics['auroc']:.4f}")
        print(f"  AUPRC:       {metrics['auprc']:.4f} ★")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['tp']:<6} FP: {metrics['fp']:<6}")
        print(f"  FN: {metrics['fn']:<6} TN: {metrics['tn']:<6}")
        print(f"{'='*80}\n")
        
        scheduler.step(metrics['auprc'])
        
        # Save best model
        if metrics['auprc'] > best_auprc:
            best_auprc = metrics['auprc']
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'X_mean': X_mean,
                'X_std': X_std
            }, 'best_sepsis_model.pt')
            
            print(f"  ★ New best AUPRC! Model saved.\n")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best AUPRC: {best_auprc:.4f} at epoch {best_epoch}")
    
    return model

if __name__ == "__main__":
    train_transformer(
        raw_dir=r"D:\college\PROJECTS-SEM 4\mimic project\federated_clients",
        augmented_dir=r"D:\college\PROJECTS-SEM 4\mimic project\federated_clients\augmented",
        n_epochs=30,
        batch_size=64,
        lr=1e-4
    )