import torch
from transformer_model import SepsisTransformer
from train_transformer import load_all_clients, compute_metrics
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def evaluate_best_model(
    checkpoint_path='best_transformer_model.pt',
    data_dir=r"D:\college\PROJECTS-SEM 4\mimic project\federated_clients\augmented",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load and evaluate the best trained model
    """
    
    print("Loading best model...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Load model
    model = SepsisTransformer(
        n_features=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']+1})")
    print(f"Training metrics:")
    for key, val in checkpoint['metrics'].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
    
    # Load validation data
    print("\nLoading validation data...")
    _, _, X_val, y_val = load_all_clients(data_dir)
    
    # Normalize
    X_val_norm = (X_val - X_mean) / X_std
    
    # Create dataloader
    val_dataset = TensorDataset(X_val_norm, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Evaluate
    print("\nEvaluating on validation set...")
    all_y_true = []
    all_y_pred_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_x = batch_x[:, :, [0, 1, 2, 3, 5]]
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_y_true.extend(batch_y.numpy())
            all_y_pred_probs.append(probs)
    
    all_y_true = np.array(all_y_true)
    all_y_pred_probs = np.vstack(all_y_pred_probs)
    
    # Compute final metrics
    metrics = compute_metrics(all_y_true, all_y_pred_probs)
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print("="*80)
    
    # Plot ROC and PR curves
    plot_curves(all_y_true, all_y_pred_probs[:, 1], metrics)
    
    return model, metrics


def plot_curves(y_true, y_scores, metrics):
    """
    Plot ROC and Precision-Recall curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax1.plot(fpr, tpr, label=f'AUROC = {metrics["auroc"]:.4f}', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ax2.plot(recall, precision, label=f'AUPRC = {metrics["auprc"]:.4f}', linewidth=2, color='orange')
    ax2.axhline(y=metrics['precision'], color='gray', linestyle='--', label='Baseline')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Curves saved to: transformer_evaluation.png")
    plt.show()


if __name__ == "__main__":
    evaluate_best_model()