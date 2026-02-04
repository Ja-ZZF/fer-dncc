import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class FERPlusDataset(Dataset):
    """
    FER+ (Facial Expression Recognition Plus) Dataset loader with preprocessing.
    
    Data Preprocessing Steps:
    1. Resizing and Standardization: Images are kept at 48×48 resolution
    2. Normalization: Pixel values are normalized by dividing by 255 (range [0, 1])
    3. Grayscale Conversion: YUV/RGB images are converted to grayscale for better feature
       relationships and reduced computational complexity
    4. Label Encoding: One-hot encoded labels for emotion classification
    5. Data Augmentation: Random horizontal flips during training for better generalization
    """
    
    def __init__(self, csv_file, usage='Training', transform=None, label_type='hard', augment=True):
        """
        Args:
            csv_file (str): Path to fer2013plus.csv
            usage (str): 'Training', 'PublicTest', or 'PrivateTest'
            transform (callable, optional): Additional transforms to apply
            label_type (str): 'soft' for 8-dim probability vector, 'hard' for argmax class index
            augment (bool): Enable data augmentation (only for training set)
        """
        self.transform = transform
        self.label_type = label_type
        self.augment = augment and (usage == 'Training')  # Only augment training data
        
        df = pd.read_csv(csv_file)
        self.data_frame = df[df['Usage'] == usage].reset_index(drop=True)
        
        # FER+ emotion columns: from 'neutral' to 'contempt' (8 classes)
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness',
                                'anger', 'disgust', 'fear', 'contempt']
        
        self.usage = usage
        print(f"Loaded {len(self.data_frame)} {usage} samples")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # ============ Step 1: Load and Resize ============
        # Load pixel values from CSV (already 48×48)
        pixels_str = self.data_frame.loc[idx, 'pixels']
        pixels = np.array(pixels_str.split(), dtype=np.uint8)
        image = pixels.reshape(48, 48)  # Shape: (48, 48), Grayscale
        
        # ============ Step 2: Load Labels ============
        # Load emotion votes (8 values)
        emotion_votes = self.data_frame.loc[idx, self.emotion_columns].values.astype(np.float32)
        
        # Convert to probability distribution (normalize sum to 1)
        total = emotion_votes.sum()
        if total > 0:
            emotion_probs = emotion_votes / total
        else:
            # Fallback: uniform distribution (should not happen in clean data)
            emotion_probs = np.ones(8, dtype=np.float32) / 8
        
        # Encode labels
        if self.label_type == 'hard':
            # Hard label: argmax of probability distribution
            label = np.argmax(emotion_probs).astype(np.int64)
        else:
            # Soft label: one-hot encoded probability vector
            label = emotion_probs  # shape (8,)
        
        # ============ Step 3: Data Augmentation (Random Horizontal Flip) ============
        if self.augment and np.random.rand() > 0.5:
            # Horizontal flip for training data augmentation
            image = np.fliplr(image).copy()
        
        # ============ Step 4: Normalization (Convert to Tensor) ============
        # Apply additional transforms if provided
        if self.transform:
            # Convert numpy array to PIL Image for transform compatibility
            image = Image.fromarray(image).convert('L')  # 'L' = Grayscale
            image = self.transform(image)
        else:
            # Manual normalization: pixel values [0, 255] -> [0, 1]
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        # Convert label to tensor
        if self.label_type == 'hard':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.from_numpy(label).float()
        
        return image, label


def get_data_loaders(csv_path, batch_size=32, num_workers=4, label_type='hard'):
    """
    Create DataLoaders for training, validation, and testing.
    
    Preprocessing Pipeline:
    - Input size: 48×48 grayscale images
    - Normalization: Pixel values divided by 255 (range [0, 1])
    - Augmentation: Random horizontal flips for training set only
    
    Args:
        csv_path (str): Path to fer2013plus.csv
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        label_type (str): 'hard' -> class index (0-7); 'soft' -> probability vector
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transforms for preprocessing
    # ToTensor() automatically converts PIL Image to [0, 1] range
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Standardization
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets with data augmentation enabled only for training
    train_dataset = FERPlusDataset(
        csv_path, 
        usage='Training', 
        transform=train_transform, 
        label_type=label_type,
        augment=True  # Enable horizontal flip augmentation
    )
    
    val_dataset = FERPlusDataset(
        csv_path, 
        usage='PrivateTest', 
        transform=test_transform, 
        label_type=label_type,
        augment=False  # No augmentation for validation
    )
    
    test_dataset = FERPlusDataset(
        csv_path, 
        usage='PublicTest', 
        transform=test_transform, 
        label_type=label_type,
        augment=False  # No augmentation for testing
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(
        './fer2013/fer2013plus.csv',
        batch_size=64,
        label_type='hard'  # or 'soft'
    )
    
    # Quick test
    for images, labels in train_loader:
        print("Image shape:", images.shape)   # e.g., [64, 1, 48, 48]
        print("Label shape:", labels.shape)   # hard: [64]; soft: [64, 8]
        print("Sample label:", labels[0])
        break