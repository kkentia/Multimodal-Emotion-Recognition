# TRAINING THE MEDIAPIPE MODEL WITH THE CSV FILE --> MULTILAYER PERCEPTRON
# THIS FACE MESH MODEL IS TRAINED ON STATIC IMAGES
# Epoch[60/60] - Loss: 0.8534 - Train Acc: 66.02% - Val Acc: 64.32%
# Epoch[100/100] - Loss: 0.7975 - Train Acc: 68.52% - Val Acc: 63.42%

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 1. SETUP & PATHS
# ==========================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(REPO_ROOT, "data", "face_mesh_dataset.csv")
SAVE_DIR = os.path.join(REPO_ROOT, "models", "saved_weights")
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 5
INPUT_FEATURES = 478 * 3  # 1434 coordinates

# CPU is actually faster than GPU for tiny tabular datasets!
device = torch.device("cpu")
print(f"🖥️ Using device: {device}")

# ==========================================
# 2. CUSTOM PYTORCH DATASET
# ==========================================
class FaceMeshDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. BUILD THE NEURAL NETWORK (MLP)
# ==========================================
class FaceMeshMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaceMeshMLP, self).__init__()
        
        # A lightweight, 4-layer Deep Neural Network
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512), # Normalizes data to prevent exploding gradients
            nn.ReLU(),
            nn.Dropout(0.3),     # Randomly turns off 30% of neurons to prevent overfitting

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, num_classes) # Final output layer (5 emotions)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. DATA LOADING & SPLITTING
# ==========================================
if __name__ == '__main__':
    print(f"\n📂 Loading dataset from {CSV_PATH}...")
    
    # Load the CSV using Pandas
    df = pd.read_csv(CSV_PATH)
    
    # Split the data: 'label' is what we want to predict (y), the rest are coordinates (X)
    y = df['label'].values
    X = df.drop(columns=['label']).values

    # Randomly split into 80% Training and 20% Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

    train_dataset = FaceMeshDataset(X_train, y_train)
    val_dataset = FaceMeshDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 5. INITIALIZE MODEL & TRAINING TOOLS
    # ==========================================
    print("\n🧠 Building Face Mesh MLP...")
    model = FaceMeshMLP(input_size=INPUT_FEATURES, num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 6. THE TRAINING LOOP
    # ==========================================
    best_val_acc = 0.0

    print("\n🚀 Starting Training...")
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
        
        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch[{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "best_mesh_model.pth") # saves in models/saved_weights
            torch.save(model.state_dict(), save_path)
            print(f"   ⭐ New best model saved!")

    print(f"\n🎉 Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")