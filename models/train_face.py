import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "faces")
SAVE_DIR = os.path.join(REPO_ROOT, "models", "saved_weights")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64 #is 1-dimensional; batch_size=labels.size(0) so e.g. [0,5,3,2...] where 0 is angry etc.
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 6 # Angry, Disgust, Fear, Happy, Neutral, Sad

#select available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except (ImportError,RuntimeError):
        device = torch.device("cpu")

print(f"using: {device}")

# DATA PREPARATION

# formatting for ResNet #https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
transform = transforms.Compose([
    transforms.Resize((224, 224)), #img resize
    #data augmentation
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),     
    transforms.ToTensor(), #convert pixels to 0.0-1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalization
])


if __name__ == '__main__':
    print("\nLoading dataset...")
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    print(f"{len(full_dataset)} total images belonging to {full_dataset.classes}")

    # split dataset 80train-20validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    # TRANSFER LEARNING MDOEL

    print("\nBuilding ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #load pretrained ResNet

    # freeze the pre-trained layers (do not do calculus)
    for param in model.parameters():
        param.requires_grad = False

    #replace the final classification layer for our 6 emotions
    num_ftrs = model.fc.in_features #in_features: input features. by default, requires_grad = True
    #model.fc is the last layer of the ResNet model. We are overwriting it:
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # last layer is simple matrix multiplication; takes 512 ResNet params and connects them to 1 of 6 emotions --> Linear Classifier

    model = model.to(device)

    # Loss func and Optimizer (for the final layer)
    criterion = nn.CrossEntropyLoss() #compares the model guesses(logits) to the true labels
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)#optimize last layer


    #TRAINING
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # train
        model.train()
        current_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()      #clear old grads, otherwise math from batch2 adds to batch1
            outputs = model(inputs)    # forward pass
            loss = criterion(outputs, labels) #calc error
            loss.backward()            # backprop & chainrule: calc gradient
            optimizer.step()           #update weights: takes the grads from .backward()
            
            current_loss += loss.item() #loss is a tensor
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) #batch size; counts the nb of img we tested
            correct += (predicted == labels).sum().item() #sum counts and item turns into integer
            
        train_acc = 100 * correct / total
        
        # VALIDATION 
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # no grad calc during validation, only testing
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)#forwardpass, output is raw logits
                _, predicted = torch.max(outputs.data, 1) #torch.max returns a value and index (corresponding to an emotion)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {current_loss/len(train_loader):.4f} - train Acc: {train_acc:.2f}% - val Acc: {val_acc:.2f}%")
        
        # save model if its the best  so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "best_face_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}!")

    print("\ntraining done")