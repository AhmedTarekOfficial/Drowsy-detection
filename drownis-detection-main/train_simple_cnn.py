# ============================================================
#  train_simple_cnn.py  —  Training Pipeline for SimpleFatigueCNN
#  Usage:  python train_simple_cnn.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm

from config import *
from simple_cnn import build_simple_model, get_device


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class FatigueDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Reads from:
            root_dir/drowsy/
            root_dir/undrowsy/
        """
        self.samples   = []
        self.transform = transform

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Folder not found: {class_dir}\n"
                    f"Expected: {CLASS_NAMES} inside {root_dir}"
                )
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"  Total samples loaded: {len(self.samples)}")
        for i, name in enumerate(CLASS_NAMES):
            count = sum(1 for _, l in self.samples if l == i)
            print(f"  {name}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────
# TRANSFORMS — Using simple normalization (NOT ImageNet stats
# since this CNN is trained from scratch)
# ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],    # Simple centering
                         [0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="  Training", leave=False,
                bar_format='{l_bar}{bar:30}{r_bar}')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        # Show live loss & accuracy in the progress bar
        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="  Validating", leave=False,
                bar_format='{l_bar}{bar:30}{r_bar}')
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs     = model(imgs)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                             acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────
def run_training(model, train_loader, val_loader, device,
                 epochs=EPOCHS, lr=LEARNING_RATE):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    save_path = os.path.join(MODELS_DIR, "simple_cnn_fatigue.pth")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.3f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.3f}")

        # Save best model
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), save_path)
            print(f"    ★ Best model saved → {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("  Early stopping triggered.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"\n  ✓ Best model loaded from → {save_path}")
    return model, history


# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────
def final_evaluation(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    ax.set_title("Simple CNN — Confusion Matrix")
    path = os.path.join(LOGS_DIR, "simple_cnn_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {path}")

    return all_preds, all_labels


# ──────────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ──────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Simple CNN — Training History", fontsize=14, fontweight='bold')

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], 'b-', label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], 'r-', label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(LOGS_DIR, "simple_cnn_training_history.png")
    plt.savefig(path, dpi=150)
    print(f"  ✓ Training plot saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()

    # ── Load dataset ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  SIMPLE CNN — Training from Scratch")
    print("="*60)

    print("\n=== Loading Dataset ===")
    full_dataset = FatigueDataset(DATA_DIR, transform=train_transform)
    n = len(full_dataset)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_test - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Val and test use no augmentation
    val_ds.dataset  = FatigueDataset(DATA_DIR, transform=val_transform)
    test_ds.dataset = FatigueDataset(DATA_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

    # ── Build model from scratch ──────────────────────────────
    print("\n=== Building Simple CNN from Scratch ===")
    model = build_simple_model(num_classes=2).to(device)

    # ── Train ─────────────────────────────────────────────────
    print("\n=== Training ===")
    model, history = run_training(
        model, train_loader, val_loader, device,
        epochs=5, lr=1e-3  # Fast training — 5 epochs only
    )

    # ── Evaluate ──────────────────────────────────────────────
    print("\n=== Final Evaluation on Test Set ===")
    final_evaluation(model, test_loader, device)
    plot_history(history)

    print("\n" + "="*60)
    print("  ✓ Simple CNN training complete!")
    print(f"  Model saved → saved_models/simple_cnn_fatigue.pth")
    print("="*60)
