import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
# She needs to point this to the folder containing her 4 emotion subfolders
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mesh_images'))
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_weights'))
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005 # A slightly lower LR is often better for fine-tuning
NUM_CLASSES = 4 # Angry, Fearful, Happy, Sad

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
def get_squeezenet_model(num_classes):
    """Loads pre-trained SqueezeNet and adapts it for our specific emotion classes."""
    # Load SqueezeNet v1.1 (lighter and faster for webcams)
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    
    # Swap out the final classification layer
    model.classifier[1] = nn.Conv2d(
        in_channels=512, 
        out_channels=num_classes, 
        kernel_size=(1, 1), 
        stride=(1, 1)
    )
    model.num_classes = num_classes
    return model

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
if __name__ == '__main__':
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Define how the images should be processed before entering the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # SqueezeNet expects 224x224
        transforms.RandomHorizontalFlip(), # Basic data augmentation to prevent overfitting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the entire dataset using ImageFolder
    # It automatically reads the folder names as the class labels!
    print(f"Loading images from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # Split into 80% Training and 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders to feed the model in batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = get_squeezenet_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    # START MODEL TRAINING
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # VALIDATION PHASE
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
        
        # Save the best model weights
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "best_squeezenet_mesh.pth")
            torch.save(model.state_dict(), save_path)
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    print(f"Complete! Max Validation Accuracy: {best_acc:.2f}%")