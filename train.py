import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from data_loader import get_data_loaders
from torchvision.models import resnet18, resnet34, resnet50


# ======================== ResNet Network Model ========================
class EmotionResNet(nn.Module):
    """
    ResNet-based network for facial expression recognition (FER).
    Adapts pre-trained ResNet to handle grayscale 48x48 images and 8-class emotion classification.
    """
    def __init__(self, num_classes=8, resnet_type='resnet18', pretrained=False):
        """
        Args:
            num_classes (int): Number of emotion classes (default: 8)
            resnet_type (str): Type of ResNet ('resnet18', 'resnet34', 'resnet50')
            pretrained (bool): Whether to use pretrained weights from ImageNet
        """
        super(EmotionResNet, self).__init__()
        
        # Load ResNet model
        if resnet_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")
        
        # Modify first convolutional layer to accept grayscale (1 channel) input
        # Original ResNet expects 3 channels (RGB)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,  # grayscale input
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Replace the final fully connected layer for emotion classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# ======================== CNN Network Model ========================
class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network for facial expression recognition (FER).
    Input: grayscale images of size 48x48
    Output: 8-class emotion classification
    Architecture based on DCNN with 64->128->256 channels
    """
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()
        
        # Block 1: 64 filters
        # Input: (None, 1, 48, 48)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # (None, 64, 48, 48)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # (None, 64, 48, 48)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)  # (None, 64, 16, 16)
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2: 128 filters
        # Input: (None, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (None, 128, 16, 16)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # (None, 128, 16, 16)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (None, 128, 8, 8)
        self.dropout2 = nn.Dropout(0.25)
        
        # Block 3: 256 filters
        # Input: (None, 128, 8, 8)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (None, 256, 8, 8)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # (None, 256, 8, 8)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (None, 256, 4, 4)
        self.dropout3 = nn.Dropout(0.25)
        
        # Flatten: 256 * 4 * 4 = 4096
        # Fully Connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 128)  # (None, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, num_classes)  # (None, 8)
        
    def forward(self, x):
        # Block 1: 48x48 -> 16x16 (stride=3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2: 16x16 -> 8x8
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3: 8x8 -> 4x4
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu1(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        return x


# ======================== Training Function ========================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ======================== Validation Function ========================
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ======================== Test Function ========================
def test(model, test_loader, device):
    """Test the model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


# ======================== Main Training Function ========================
def main():
    # Hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    num_classes = 8
    csv_path = './fer2013/fer2013plus.csv'
    checkpoint_dir = './checkpoints'
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        csv_path,
        batch_size=batch_size,
        num_workers=4,
        label_type='hard'  # Use hard labels for classification
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Model setup
    print("\nInitializing model...")
    # Choose model: 'cnn' or 'resnet' (resnet18, resnet34, resnet50)
    model_type = 'resnet18'  # Change to 'resnet34', 'resnet50', or 'cnn'
    
    if model_type == 'cnn':
        model = EmotionCNN(num_classes=num_classes)
    elif model_type.startswith('resnet'):
        model = EmotionResNet(num_classes=num_classes, resnet_type=model_type, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    print(f"Model: {model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("\nStarting training...\n")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print()
    
    # Load best model and test
    print("Loading best model...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nTesting...")
    test_acc = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "="*50)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
