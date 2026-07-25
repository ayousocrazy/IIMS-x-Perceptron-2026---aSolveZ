import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset import MedicineDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    IMAGES_DIR = "./images"
    EPOCHS = 25
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    MODEL_OUT = "model.pth"
    CLASSES_OUT = "classes.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Standard training transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load 100% of the training dataset
    train_dataset = MedicineDataset(IMAGES_DIR, folder_based=True, transform=train_transform)

    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Dataset loaded: {len(train_dataset)} images across {num_classes} classes")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ResNet-18 with frozen backbone
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    # Swap the classifier head to match class count
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save artifacts
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "arch": "resnet18"
    }, MODEL_OUT)

    train_dataset.save_classes(CLASSES_OUT)

    print("\nDone!")
    print(f"Saved weights to '{MODEL_OUT}' and class labels to '{CLASSES_OUT}'.")


if __name__ == "__main__":
    main()