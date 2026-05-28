#train_static_mesh.py
# Epoch [100/100] | Val Acc: 54.08% probably bcz it struggles with 'digust' emotion
# without disgust: Epoch [100/100] | Val Acc: 54.98% --> nope, that wasnt it.
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(REPO_ROOT, "data", "static_mesh_dataset.csv")
SAVE_DIR = os.path.join(REPO_ROOT, "models", "saved_weights")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
INPUT_FEATURES = 10
NUM_CLASSES = 6 # happy, sad, fear, angry, neutral, disgust ( maybe delete it)

class StaticDataset(Dataset):  # we fill feed it the .csv ds in the form of a pandas Numpy Array
    def __init__(self, X, y): # X is the data, y is the labels
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
    
    #this class take the numpy arrays and converts them into PyTorch tensors (=matrices, these can be then sent to device to do calculus really fast)

class StaticMLP(nn.Module): #nn.module is base class for all nn modules: ur models should subclass this class 

    def __init__(self, input_size, num_classes):
        super(StaticMLP, self).__init__()
        self.network = nn.Sequential( #it is a sequential container
            nn.Linear(input_size, 128), #removed
            #nn.BatchNorm1d(64), #normalise 2d and 3d data
            nn.LayerNorm(128),
            nn.ReLU(), #non linear activation func
            nn.Dropout(0.3), #regularization technique that randomly set 20% of input to 0 during training, preventing nodes from co-adapting
            nn.Linear(128, 64), #input size is last output size
            #nn.BatchNorm1d(32), removed
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p= 0.3), #p is probability of an elem to be zero'ed
            nn.Linear(64, num_classes) #output is the number of features / labels we want to classify
        )
    def forward(self, x): return self.network(x)


# MAIN.APP 

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=['label']).values # the data on which we train does not have the labels, Y does
    y = df['label'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_loader = DataLoader(StaticDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StaticDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = StaticMLP(INPUT_FEATURES, NUM_CLASSES)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight= weights_tensor) #loss func
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  #optim from torch as well
    
    best_acc = 0.0

    # START MODEL TRAINING
    print("starting training")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad() # reset the grads of all optimized tensors to 0 (why?)
            loss = criterion(model(inputs), labels) # 1. calc error
            loss.backward() # 2. fix the error with calculus
            optimizer.step() # 3. update weights
            #--> as long as these 3 lines exits, the model is training
            
        #VALIDATION
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_static_mesh.pth"))
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Val Acc: {val_acc:.2f}%")

    print(f"Complete! Max Validation Accuracy: {best_acc:.2f}%")