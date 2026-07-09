import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

def create_federated_clients(
    input_csv='mimiciv_processed.csv',
    output_dir='federated_clients',
    n_clients=100,
    train_clients=80,
    alpha=0.5
):
    """
    Split MIMIC-IV data into federated clients with patient-level separation
    
    Args:
        input_csv: Preprocessed MIMIC-IV data with subject_id
        output_dir: Directory to save client files
        n_clients: Total number of clients
        train_clients: Number of training clients (0 to train_clients-1)
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
    """
    
    print("="*80)
    print("FEDERATED CLIENT PARTITIONING")
    print("="*80)
    
    # Load data
    print("\n1. Loading preprocessed data...")
    df = pd.read_csv(r"D:\college\PROJECTS-SEM 4\mimic project\00_data\mimiciv_processed.csv")
    
    print(f"   Total records: {len(df):,}")
    print(f"   ICU stays: {df['stay_id'].nunique():,}")
    print(f"   Patients: {df['subject_id'].nunique():,}")
    
    # Get unique patients
    unique_patients = df['subject_id'].unique()
    n_patients = len(unique_patients)
    
    print(f"\n2. Partitioning {n_patients:,} patients across {n_clients} clients...")
    
    # Shuffle patients
    np.random.seed(42)
    shuffled_patients = np.random.permutation(unique_patients)
    
    # Dirichlet distribution for heterogeneous client sizes
    proportions = np.random.dirichlet([alpha] * n_clients)
    client_sizes = (proportions * n_patients).astype(int)
    client_sizes[-1] += n_patients - client_sizes.sum()
    
    # Assign patients to clients
    patient_assignments = {}
    idx = 0
    
    for client_id in range(n_clients):
        client_patients = shuffled_patients[idx:idx + client_sizes[client_id]]
        for patient in client_patients:
            patient_assignments[patient] = client_id
        idx += client_sizes[client_id]
    
    assert len(patient_assignments) == n_patients
    
    print(f"   Patient assignment complete")
    print(f"   Client size range: {client_sizes.min()} to {client_sizes.max()} patients")
    
    # Map to dataframe
    df['client_id'] = df['subject_id'].map(patient_assignments)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process each ICU stay
    print(f"\n3. Creating time-series tensors...")
    
    client_data = {i: {'X': [], 'y': []} for i in range(n_clients)}
    
    for stay_id, group in tqdm(df.groupby('stay_id'), desc="Processing stays"):
        group = group.sort_values('chart_hour')
        
        # Extract features (5 features: HR, SBP, DBP, SpO2, Shock Index)
        features = group[['heart_rate', 'sbp', 'dbp', 'spo2', 'shock_index']].values
        
        # Standardize to 24 hours
        if len(features) < 24:
            padding = np.repeat(features[-1:], 24 - len(features), axis=0)
            features = np.vstack([features, padding])
        else:
            features = features[:24]
        
        label = group['mortality'].iloc[0]
        client_id = group['client_id'].iloc[0]
        
        client_data[client_id]['X'].append(features)
        client_data[client_id]['y'].append(label)
    
    # Save clients
    print(f"\n4. Saving {n_clients} client files...")
    
    train_stats = []
    test_stats = []
    
    for client_id in tqdm(range(n_clients), desc="Writing files"):
        if len(client_data[client_id]['X']) == 0:
            continue
        
        X = torch.tensor(np.array(client_data[client_id]['X']), dtype=torch.float32)
        y = torch.tensor(client_data[client_id]['y'], dtype=torch.long)
        
        output_file = output_path / f"client_{client_id}.pt"
        torch.save({'X': X, 'y': y}, output_file)
        
        # Statistics
        n_sepsis = (y == 1).sum().item()
        stats = {
            'client_id': client_id,
            'n_samples': len(y),
            'n_sepsis': n_sepsis,
            'sepsis_ratio': n_sepsis / len(y)
        }
        
        if client_id < train_clients:
            train_stats.append(stats)
        else:
            test_stats.append(stats)
    
    # Summary
    print("\n" + "="*80)
    print("PARTITIONING SUMMARY")
    print("="*80)
    
    train_df = pd.DataFrame(train_stats)
    test_df = pd.DataFrame(test_stats)
    
    print(f"\nTRAINING SET (Clients 0-{train_clients-1}):")
    print(f"  Total samples: {train_df['n_samples'].sum():,}")
    print(f"  Sepsis cases: {train_df['n_sepsis'].sum():,} ({100*train_df['n_sepsis'].sum()/train_df['n_samples'].sum():.2f}%)")
    
    print(f"\nTEST SET (Clients {train_clients}-{n_clients-1}):")
    print(f"  Total samples: {test_df['n_samples'].sum():,}")
    print(f"  Sepsis cases: {test_df['n_sepsis'].sum():,} ({100*test_df['n_sepsis'].sum()/test_df['n_samples'].sum():.2f}%)")
    
    # Leakage verification
    print("\n" + "="*80)
    print("PATIENT LEAKAGE VERIFICATION")
    print("="*80)
    
    train_patients = set(df[df['client_id'] < train_clients]['subject_id'].unique())
    test_patients = set(df[df['client_id'] >= train_clients]['subject_id'].unique())
    overlap = train_patients & test_patients
    
    if len(overlap) == 0:
        print(f"✓ VERIFICATION PASSED")
        print(f"  Training patients: {len(train_patients):,}")
        print(f"  Test patients: {len(test_patients):,}")
        print(f"  Overlap: 0")
    else:
        print(f"✗ VERIFICATION FAILED")
        print(f"  {len(overlap)} patients in both sets")
        return False
    
    print(f"\n✓ Client files saved to: {output_path}")
    return True


if __name__ == "__main__":
    success = create_federated_clients(
        input_csv='r"D:\college\PROJECTS-SEM 4\mimic project\00_data\mimiciv_processed.csv"',
        output_dir='federated_clients',
        n_clients=100,
        train_clients=80
    )
    
    if success:
        print("\n" + "="*80)
        print("READY FOR TRAINING")
        print("="*80)