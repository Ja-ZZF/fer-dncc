import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import numpy as np

class FERPlusDataset(Dataset):
    def __init__(self, csv_file, usage='Training', transform=None, label_type='soft'):
        """
        Args:
            csv_file (str): Path to ferplus.csv
            usage (str): 'Training', 'PublicTest', or 'PrivateTest'
            transform (callable, optional): Transform on image (expects PIL or tensor)
            label_type (str): 'soft' for 8-dim probability vector, 'hard' for argmax class index
        """
        self.transform = transform
        self.label_type = label_type
        df = pd.read_csv(csv_file)
        self.data_frame = df[df['Usage'] == usage].reset_index(drop=True)
        # FER+ emotion columns: from 'neutral' to 'contempt' (8 columns)
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness',
                                'anger', 'disgust', 'fear', 'contempt']

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load pixels
        pixels_str = self.data_frame.loc[idx, 'pixels']
        pixels = np.array(pixels_str.split(), dtype=np.uint8)
        image = pixels.reshape(48, 48)  # H x W, uint8

        # Load emotion votes (8 values)
        emotion_votes = self.data_frame.loc[idx, self.emotion_columns].values.astype(np.float32)
        
        # Convert to probability distribution (normalize sum to 1)
        total = emotion_votes.sum()
        if total > 0:
            emotion_probs = emotion_votes / total
        else:
            # Fallback: uniform or zero (should not happen in clean data)
            emotion_probs = np.ones(8, dtype=np.float32) / 8

        if self.label_type == 'hard':
            label = np.argmax(emotion_probs).astype(np.int64)
        else:  # 'soft'
            label = emotion_probs  # shape (8,)

        # Apply transform (note: ToTensor() expects numpy array in [0,255] with dtype uint8)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0  # Add channel dim

        if self.label_type == 'hard':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.from_numpy(label).float()

        return image, label


def get_data_loaders(csv_path, batch_size=32, num_workers=4, label_type='hard'):
    """
    Returns train/val/test DataLoaders.
    label_type: 'hard' -> integer class index (0-7); 'soft' -> 8-dim probability vector
    """
    # Note: transforms.ToTensor() converts [H, W] uint8 numpy -> [1, H, W] float tensor in [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # You can add Normalize(mean=[0.5], std=[0.5]) if needed
    ])

    train_dataset = FERPlusDataset(csv_path, usage='Training', transform=transform, label_type=label_type)
    val_dataset = FERPlusDataset(csv_path, usage='PrivateTest', transform=transform, label_type=label_type)
    test_dataset = FERPlusDataset(csv_path, usage='PublicTest', transform=transform, label_type=label_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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