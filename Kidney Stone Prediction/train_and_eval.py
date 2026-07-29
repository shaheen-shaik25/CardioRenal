import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from models.xresnet50 import get_xresnet50
from models.benchmark_models import get_resnet50, get_mobilenetv2, get_vgg16, get_inceptionv3
from utils.preprocessing import get_transforms
from utils.metrics import calculate_metrics
from generate_dataset import create_dataset

def train_model(model, train_loader, criterion, optimizer, device, epochs=3, is_inception=False):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            if is_inception:
                inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    return model

def evaluate_model(model, test_loader, device, is_inception=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if is_inception:
                inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    return metrics, y_true, y_pred, y_probs

def run_training_pipeline():
    base_dir = os.path.abspath("dataset")
    if not os.path.exists(os.path.join(base_dir, "train")):
        print("Dataset not found. Generating synthetic CT scan dataset...")
        create_dataset(base_dir, train_count=100, test_count=25)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Data Loaders
    train_transform = get_transforms(augment=True)
    test_transform = get_transforms(augment=False)

    train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    weights_dir = os.path.abspath(os.path.join("models", "weights"))
    os.makedirs(weights_dir, exist_ok=True)

    models_to_train = {
        'XResNet50': get_xresnet50(num_classes=2),
        'ResNet50': get_resnet50(num_classes=2),
        'MobileNetV2': get_mobilenetv2(num_classes=2),
        'VGG16': get_vgg16(num_classes=2),
        'InceptionV3': get_inceptionv3(num_classes=2)
    }

    metrics_summary = {}

    for name, model in models_to_train.items():
        print(f"\n--- Training {name} ---")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        is_inc = (name == 'InceptionV3')
        trained_model = train_model(model, train_loader, criterion, optimizer, device, epochs=3, is_inception=is_inc)

        # Save weights
        save_path = os.path.join(weights_dir, f"{name.lower()}_best.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f"Saved weights to {save_path}")

        # Evaluate
        print(f"Evaluating {name} on test dataset...")
        metrics, _, _, _ = evaluate_model(trained_model, test_loader, device, is_inception=is_inc)
        metrics_summary[name] = metrics
        print(f"  {name} Accuracy: {metrics['accuracy']*100:.2f}%, F1-Score: {metrics['f1_score']:.4f}")

    # Save metrics summary JSON
    summary_path = os.path.join(weights_dir, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nMetrics summary saved to {summary_path}")

if __name__ == "__main__":
    run_training_pipeline()
