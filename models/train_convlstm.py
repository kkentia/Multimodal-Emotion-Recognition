import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(REPO_ROOT, "data", "convlstm_dataset.csv")
SAVE_DIR = os.path.join(REPO_ROOT, "models", "saved_weights")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
SEQ_LENGTH = 10
FEATURES = 10
NUM_CLASSES = 5

device = torch.device("cpu")

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        # Reshape into 3D Temporal Matrix: (Batch, TimeSteps, Features)
        X_3d = X.reshape(-1, SEQ_LENGTH, FEATURES)
        self.X = torch.tensor(X_3d, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ConvLSTM1D(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes):
        super(ConvLSTM1D, self).__init__()
        # 1D Conv filters out talking noise, focuses on smooth emotion
        self.conv1d = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        # LSTM tracks the timeline of the muscle movement
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x is (Batch, Time, Features). Conv1D needs (Batch, Features, Time)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1d(x)))
        x = x.transpose(1, 2) # Swap back for LSTM
        
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :] # Grab the memory at the final frame
        
        out = self.relu(self.fc1(final_state))
        out = self.dropout(out)
        return self.fc2(out)

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    # ---- DEBUG ----
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("actor_id unique values:", df['actor_id'].unique())
    print("Train rows:", len(df[df['actor_id'] <= 20]))
    print("Val rows:", len(df[df['actor_id'] > 20]))
    # ---------------

    train_df = df[df['actor_id'] <= 20]    
    # Subject-Wise Split: Actors 1-20 Train, Actors 21-24 Test (No Leakage!)
    train_df = df[df['actor_id'] <= 20]
    val_df = df[df['actor_id'] > 20]
    
    y_train = train_df['label'].values
    X_train = train_df.drop(columns=['actor_id', 'label']).values
    y_val = val_df['label'].values
    X_val = val_df.drop(columns=['actor_id', 'label']).values

    # ---- DEBUG: find bad columns ----
    feature_df = train_df.drop(columns=['actor_id', 'label'])
    for col in feature_df.columns:
        non_numeric = pd.to_numeric(feature_df[col], errors='coerce').isna().sum()
        if non_numeric > 0:
            print(f"BAD COLUMN: {col} has {non_numeric} non-numeric rows")
            print(feature_df[col][pd.to_numeric(feature_df[col], errors='coerce').isna()].unique())
    # ---------------------------------

    X_train = np.nan_to_num(X_train.astype(np.float32), nan=0.0)
    X_val   = np.nan_to_num(X_val.astype(np.float32),   nan=0.0)
    y_train = y_train.astype(np.int64)
    y_val   = y_val.astype(np.int64)

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ConvLSTM1D(input_features=FEATURES, hidden_size=64, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_acc = 0.0
    print("🚀 Starting ConvLSTM1D Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_convlstm_model.pth"))
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch[{epoch+1}/{EPOCHS}] | Val Acc: {val_acc:.2f}%")

    print(f"🎉 Complete! Max Real-World Accuracy: {best_acc:.2f}%")