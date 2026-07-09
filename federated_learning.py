import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# ===================== CONFIG =====================

NUM_CLIENTS = 100
CLIENT_SAMPLE_RATE = 0.10
COMMUNICATION_ROUNDS = 100
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 24
NUM_CLASSES = 2

RESULT_DIR = r"D:\college\PROJECTS-SEM 4\mimic project\plots"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== MODEL =====================

class GlobalHealthModel(nn.Module):
    def __init__(self, num_features, num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH):
        super().__init__()

        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        pooled_length = sequence_length // 4

        self.fc1 = nn.Linear(64 * pooled_length, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.flatten(1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)

        return x


# ===================== DATA =====================

def create_sequences(data, labels, sequence_length=SEQUENCE_LENGTH):
    sequences, seq_labels = [], []

    for i in range(0, len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
        seq_labels.append(labels[i + sequence_length - 1])

    return np.array(sequences), np.array(seq_labels)


def load_mimic_data(file_path):
    df = pd.read_csv(file_path)
    
    # 1. DROP LEAKAGE & METADATA
    # We keep subject_id for the split, then drop it
    features = [
        'heart_rate_mean', 'sbp_mean', 'dbp_mean', 'mbp_mean', 
        'resp_rate_mean', 'temperature_mean', 'spo2_mean', 'glucose_mean',
        'wbc_max', 'hemoglobin_min', 'platelet_min', 'sodium_min', 
        'potassium_max', 'bicarbonate_min', 'creatinine_max', 'bun_max', 
        'bilirubin_total_max', 'inr_max', 'pt_max', 'ptt_max', 'lactate_max',
        'shock_index', 'pulse_pressure'
    ]
    
    # 2. SEPARATE SYMBOLIC COLS (Keep for Phase B)
    symbolic_cols = ['gcs_min', 'sofa_24hours', 'aki_stage']
    
    X = df[features].fillna(df.mean()) # Neural Inputs
    y = df['sepsis_risk']              # Target
    groups = df['subject_id']          # For Grouped Split
    
    return X, y, groups, len(features)


# ===================== CLIENT SIMULATION =====================

def simulate_client_data(X_train, y_train):

    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
        y_train = y_train.numpy()

    client_data = {}

    num_samples = len(y_train)
    num_classes = len(np.unique(y_train))

    MIN_CLIENT_SAMPLES = 64

    min_size = 0

    while min_size < MIN_CLIENT_SAMPLES:

        idx_batch = [[] for _ in range(NUM_CLIENTS)]

        for k in range(num_classes):

            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(0.5, NUM_CLIENTS))

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            splits = np.split(idx_k, proportions)

            idx_batch = [idx_j + idx.tolist() 
                         for idx_j, idx in zip(idx_batch, splits)]

        min_size = min([len(idx_j) for idx_j in idx_batch])

    for client_id in range(NUM_CLIENTS):

        indices = idx_batch[client_id]

        client_data[client_id] = (
            torch.FloatTensor(X_train[indices]),
            torch.LongTensor(y_train[indices])
        )

    return client_data


# ===================== LOCAL TRAIN =====================

def local_train(model, data, labels):

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    num_samples = len(data)

    if num_samples < BATCH_SIZE:
        return model.state_dict(), 0.0

    final_loss = 0.0

    for epoch in range(LOCAL_EPOCHS):

        indices = torch.randperm(num_samples)

        for i in range(0, num_samples, BATCH_SIZE):

            batch_idx = indices[i:i+BATCH_SIZE]

            optimizer.zero_grad()
            loss = criterion(model(data[batch_idx]), labels[batch_idx])
            loss.backward()
            optimizer.step()

            if epoch == LOCAL_EPOCHS - 1:
                final_loss += loss.item()

    divisor = max(1, num_samples // BATCH_SIZE)

    return model.state_dict(), final_loss / divisor


# ===================== AGGREGATION =====================

def aggregate_models(global_model, client_models, client_weights):

    total_samples = sum(client_weights)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():

        if not torch.is_floating_point(global_dict[key]):
            continue

        global_dict[key] = torch.zeros_like(global_dict[key])

        for client_state, weight in zip(client_models, client_weights):

            if torch.is_floating_point(client_state[key]):
                global_dict[key] += (weight / total_samples) * client_state[key]

    global_model.load_state_dict(global_dict, strict=False)


# ===================== EVALUATION =====================

def evaluate_model(model, X_test, y_test):

    model.eval()

    with torch.no_grad():
        outputs = model(X_test)

        loss = nn.CrossEntropyLoss()(outputs, y_test).item()

        accuracy = (outputs.argmax(dim=1) == y_test).float().mean().item()

    return accuracy, loss


def save_roc_plot(model, X_test, y_test, path):

    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(X_test), dim=1)[:,1].numpy()

    auc = roc_auc_score(y_test.numpy(), probs)

    plt.figure()
    plt.title(f"ROC AUC = {auc:.4f}")
    plt.hist(probs, bins=50)
    plt.savefig(path)
    plt.close()

    return auc


# ===================== FEDERATED =====================

def federated_learning(X_train, y_train, X_test, y_test, num_features):

    global_model = GlobalHealthModel(num_features)

    client_data = simulate_client_data(X_train, y_train)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    history = {'rounds': [], 'accuracy': [], 'loss': []}

    for round_num in range(COMMUNICATION_ROUNDS):

        num_selected = max(1, int(NUM_CLIENTS * CLIENT_SAMPLE_RATE))

        selected_clients = np.random.choice(NUM_CLIENTS, num_selected, replace=False)

        client_models = []
        client_weights = []
        round_losses = []

        for client_id in selected_clients:

            local_model = copy.deepcopy(global_model)

            X_client, y_client = client_data[client_id]

            if len(X_client) < BATCH_SIZE:
                continue

            updated_state, loss = local_train(local_model, X_client, y_client)

            client_models.append(updated_state)
            client_weights.append(len(X_client))
            round_losses.append(loss)

        if len(client_models) == 0:
            continue

        aggregate_models(global_model, client_models, client_weights)

        if (round_num + 1) % 10 == 0:

            accuracy, test_loss = evaluate_model(
                global_model,
                X_test_tensor,
                y_test_tensor
            )

            print(f"Round {round_num+1} | Loss: {np.mean(round_losses):.4f} | Acc: {accuracy:.4f}")

            history['rounds'].append(round_num + 1)
            history['accuracy'].append(accuracy)
            history['loss'].append(test_loss)

    return global_model, history


# ===================== MAIN =====================

if __name__ == "__main__":

    train_path = r"D:\college\PROJECTS-SEM 4\mimic project\data\ml_icu_train_neurosymbolic.csv"
    test_path  = r"D:\college\PROJECTS-SEM 4\mimic project\data\ml_icu_test_neurosymbolic.csv"

    X_train, y_train, X_test, y_test, num_features = load_mimic_data(
        train_path,
        test_path
    )

    model, history = federated_learning(
        X_train, y_train, X_test, y_test, num_features
    )

    torch.save(
        model.state_dict(),
        os.path.join(RESULT_DIR, 'global_model_final.pth')
    )

    pd.DataFrame(history).to_csv(
        os.path.join(RESULT_DIR, 'training_history.csv'),
        index=False
    )

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    auc = save_roc_plot(
        model, X_test_tensor, y_test_tensor,
        os.path.join(RESULT_DIR, 'roc_curve.png')
    )

    print("\n===== FINAL =====")
    print(f"AUC: {auc:.4f}")
    print(f"Saved to: {RESULT_DIR}")
