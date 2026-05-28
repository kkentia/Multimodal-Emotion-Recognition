import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(REPO_ROOT, "data", "lstm_mesh_dataset.csv")
SAVE_DIR = os.path.join(REPO_ROOT, "models", "saved_weights")

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
SEQ_LENGTH = 5
INPUT_FEATURES = 10
NUM_CLASSES = 5

device = torch.device("cpu")

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        # Reshape X from flat (50) to 3D temporal matrix (5, 10)
        X_reshaped = X.reshape(-1, SEQ_LENGTH, INPUT_FEATURES)
        self.X = torch.tensor(X_reshaped, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True means the input tensor is (Batch, Time_Steps, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    y = df['label'].values
    X = df.drop(columns=['label']).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = EmotionLSTM(input_size=INPUT_FEATURES, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_lstm_model.pth"))

    print(f"Complete. Max Val Accuracy: {best_acc:.2f}%")