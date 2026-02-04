import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data_loader import get_data_loaders
from train import EmotionCNN


CLASS_NAMES = ['neutral', 'happiness', 'surprise', 'sadness',
               'anger', 'disgust', 'fear', 'contempt']


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, out_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close()


def infer_and_visualize(checkpoint_path, csv_path, batch_size, device, out_dir, show):
    os.makedirs(out_dir, exist_ok=True)

    # Data loaders (label_type hard for classification)
    _, _, test_loader = get_data_loaders(csv_path, batch_size=batch_size, num_workers=4, label_type='hard')

    # Model
    model = EmotionCNN(num_classes=len(CLASS_NAMES))
    map_loc = torch.device(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_loc)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(map_loc)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(map_loc)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(out_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # Plot raw confusion matrix
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, CLASS_NAMES, normalize=False, title='Confusion Matrix', out_path=cm_path)

    # Plot normalized confusion matrix
    cm_norm_path = os.path.join(out_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, CLASS_NAMES, normalize=True, title='Normalized Confusion Matrix', out_path=cm_norm_path)

    print(f"Saved: {report_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {cm_norm_path}")

    if show:
        # Also display images inline if requested
        from PIL import Image
        img = Image.open(cm_path)
        img.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FER model predictions and confusion matrix')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, default='./fer2013/fer2013plus.csv', help='Path to fer2013plus.csv')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    parser.add_argument('--out-dir', type=str, default='./visualize_outputs', help='Directory to save visualizations')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Do not open images after saving')
    parser.set_defaults(show=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer_and_visualize(args.checkpoint, args.csv, args.batch_size, args.device, args.out_dir, args.show)
