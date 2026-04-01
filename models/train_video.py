# THIS FACE MESH MODEL IS TRAINED ON DYNAMIC TIME SERIES IMAGES (VIDEO)

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

REPO_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH= os.path.join(REPO_ROOT, "data", "temporal_mesh_dataset.csv")
SAVE_DIR=os.path.join(REPO_ROOT, "models", "saved_weights")
os.makedirs(SAVE_DIR,exist_ok=True)

BATCH_SIZE= 64
EPOCHS = 100
LEARNING_RATE= 0.001

INPUT_FEATURES = 25 # 5 frames * 5 features (Left & Right Eye Aspect Ratio,Mouth aspect ratio, Left & Right eyebrow aspect ratio) per frame
NUM_CLASSES= 5

device = torch.device('cpu')
print("using device: ", device)


#PYTORCH DATASET
class TemporalDataset(Dataset): # we fill feed it the .csv ds in the form of a pandas Numpy Array
    def __init__(self, x, y): # x is the data, y is the labels
        self.x= torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x) 
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    #this class take the numpy arrays and converts them into PyTorch tensors (=matrices, these can be then sent to device to do calculus really fast)

# TEMPORAL NEURAL NETWORK
class TemporalMLP(nn.Module): #nn.module is base class for all nn modules: ur models should subclass this class 
    def __init__(self, input_size, num_classes) -> None:
        super(TemporalMLP,self).__init__() #why not just super().__init__()

        self.network = nn.Sequential( # its a sequential container
            nn.Linear(in_features=input_size, out_features=128),
            nn.BatchNorm1d(num_features=128), #applies batch normalization over a 2d or 3d input
            nn.ReLU(),
            nn.Dropout(p=0.3), #p is probability of an elem to be zero'ed

            nn.Linear(in_features=128, out_features=64), #in features here is out feats from layer from before
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(in_features=64, out_features=num_classes) # OUT_FEATURES IS NUM_CLASSES --> WE HAVE "CLASSIFIED"
        )
        
    def forward(self, x):
        return self.network(x)
    
    

# MAIN.APP : SUBJECT-WISE SPLITTING
if __name__ == '__main__':
    print("loading dataset from: ", CSV_PATH)
    df= pd.read_csv(CSV_PATH)
    
    #RAVDESS has 24 actors: we put actors 1-20 in train, 21-24 in validation
    train_df=df[df['actor_id'] <= 20]
    val_df= df[df['actor_id'] > 20]
    
    print(f"training on {len(train_df)} sequences (actors 1-20)")
    print(f"validating on {len(val_df)} sequences (actors 21-24)")
    
    #separate labels (y) and drop actor_id and label from features (X)
    y_train = train_df['label'].values
    x_train= train_df.drop(columns=['actor_id','label']).values
    
    y_val = val_df['label'].values
    x_val= val_df.drop(columns=['actor_id','label']).values
    
    train_loader = DataLoader(dataset=TemporalDataset(x_train,y_train), batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(TemporalDataset(x_val, y_val), BATCH_SIZE, shuffle=False)

    # TRAINING LOOP
    model= TemporalMLP(input_size=INPUT_FEATURES, num_classes=NUM_CLASSES).to(device) #load model to device
    criterion= nn.CrossEntropyLoss() #loss function
    optimizer= optim.Adam(model.parameters(),lr=LEARNING_RATE) #optim from torch as well
    
    best_val_acc= 0.0
    
    print ("starting training...")
    for epoch in range (EPOCHS):
        model.train()
        current_loss= 0.0
        correct= 0
        total=0
        
        for inputs, labels in train_loader:
            #first, we load them to device
            inputs, labels = inputs.to(device), labels.to(torch.long).to(device)
            optimizer.zero_grad() # reset the grads of all optimized tensors to 0 (why?)
            outputs=model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            _,predicted = torch.max(outputs.data, 1) #give one prediction with max proba from outputs
            total += labels.size(0) #the full size i think
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 *correct / total
        
        
        # validation
        model.eval()
        val_correct= 0
        val_total = 0
        
        with torch.no_grad(): #no more training when in the validation phase
            for inputs, labels in val_loader:
                inputs,labels= inputs.to(device), labels.to(torch.long).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data,1)
                val_total+= labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
            val_acc= 100 * val_correct / val_total
                

            print(f"epoch [{epoch} / {EPOCHS}] - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc =val_acc
                save_path= os.path.join(SAVE_DIR, "best_video_model.pth")
                torch.save(model.state_dict(),save_path)
                
        
        print(f"Training complete. Real validation accuracy: {best_val_acc:.2f}%")
        print(f"saved to: {save_path}")
                
        