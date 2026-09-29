import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

DATASET_DIR = "dataset"
MODELS_DIR = "models_saved"
OUTPUT_DIR = "reports_eda"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENSOR_COLS = [
    'P-PDG', 'T-PDG', 
    'P-TPT', 'T-TPT', 
    'P-MON-CKP', 'P-JUS-CKP', 
    'T-MON-CKP', 'T-JUS-CKP'
]

classes_info = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line"
}

# 1. PyTorch Dataset for Continuous Sensor Sequences
class WellTimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 2. 1D-CNN + BiLSTM Deep Learning Architecture
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, num_channels=8, num_classes=10):
        super(CNN_LSTM_Classifier, self).__init__()
        
        # 1D Convolutional Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Bidirectional LSTM for Sequential Dynamics
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Dense Classifier Head
        self.fc1 = nn.Linear(64 * 2, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (Batch, SeqLen, Channels) -> Permute to (Batch, Channels, SeqLen)
        x = x.permute(0, 2, 1)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # Permute back to (Batch, SeqLen_pooled, Channels_out) for LSTM
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        # Take last time step output
        out = lstm_out[:, -1, :]
        
        out = self.dropout(self.relu3(self.fc1(out)))
        out = self.fc2(out)
        return out

def prepare_raw_sequences(window_size=120, step_size=120):
    print("Preparing continuous raw sensor sequences for PyTorch Deep Learning...")
    sequences = []
    labels = []
    
    for c in range(10):
        folder_path = os.path.join(DATASET_DIR, str(c))
        if not os.path.exists(folder_path):
            continue
        files = glob.glob(os.path.join(folder_path, "*.parquet"))
        max_files = 10
        
        for f in files[:max_files]:
            try:
                df = pd.read_parquet(f)
                
                # Normalize pressure/temp columns to roughly 0-1 scale
                sub_df = pd.DataFrame()
                for col in SENSOR_COLS:
                    if col in df.columns:
                        s = df[col].fillna(0.0).values
                        std = np.std(s)
                        sub_df[col] = (s - np.mean(s)) / (std + 1e-6) if std > 0 else s
                    else:
                        sub_df[col] = 0.0
                        
                arr = sub_df[SENSOR_COLS].values
                n_points = len(arr)
                
                if n_points >= window_size:
                    for idx in range(0, n_points - window_size, step_size):
                        seq = arr[idx:idx + window_size]
                        sequences.append(seq)
                        labels.append(c)
            except Exception as e:
                pass
                
    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f"Prepared sequence dataset: Sequences shape={sequences.shape}, Labels shape={labels.shape}")
    return sequences, labels

def train_deep_learning_model():
    print("=" * 60)
    print(" STEP 4: PYTORCH DEEP LEARNING (1D-CNN + BiLSTM) ")
    print("=" * 60)
    
    X_raw, y_raw = prepare_raw_sequences(window_size=120, step_size=60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    
    train_dataset = WellTimeSeriesDataset(X_train, y_train)
    test_dataset = WellTimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    model = CNN_LSTM_Classifier(num_channels=8, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 15
    history = {'train_loss': [], 'val_acc': []}
    
    start_t = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for seqs, lbls in train_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seqs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Evaluation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for seqs, lbls in test_loader:
                seqs, lbls = seqs.to(device), lbls.to(device)
                outputs = model(seqs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(lbls.cpu().numpy())
                
        val_acc = accuracy_score(all_targets, all_preds)
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} - Test Accuracy: {val_acc*100:.2f}%")
        
    elapsed = time.time() - start_t
    print(f"\nPyTorch Deep Learning Model Training completed in {elapsed:.2f} seconds!")
    print(f"Final PyTorch Model Test Accuracy: {val_acc*100:.2f}%")
    
    # Save Model Weights
    torch_path = os.path.join(MODELS_DIR, "pytorch_cnn_lstm_3w.pth")
    torch.save(model.state_dict(), torch_path)
    print(f"Saved PyTorch model weights to: {torch_path}")
    
    # Plot Training Loss & Accuracy Curves
    plt.figure(figsize=(10, 4), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], color='#EF4444', marker='o')
    plt.title("PyTorch Training Loss", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [acc * 100 for acc in history['val_acc']], color='#10B981', marker='o')
    plt.title("PyTorch Validation Accuracy (%)", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "deep_learning_training_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved PyTorch training curve plot to: {curve_path}")

if __name__ == "__main__":
    train_deep_learning_model()
