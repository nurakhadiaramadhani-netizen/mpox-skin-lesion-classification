import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONFIGS
from Utils.getData import getImageLabel


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.permute(0, 3, 1, 2).to(device)
            _, label_indices = torch.max(labels, 1)
            label_indices = label_indices.to(device)

            outputs = model(images)
            loss = criterion(outputs, label_indices)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += label_indices.size(0)
            correct += predicted.eq(label_indices).sum().item()

    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def plot_training_curves(history_df, title_label, save_path):
    epochs = history_df["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history_df["train_loss"], label="Train Loss")
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        ax1.plot(epochs, history_df["val_loss"], label="Valid Loss")
    ax1.set_title(f"{title_label} - Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(epochs, history_df["train_acc"] * 100.0, label="Train Acc (%)")
    if "val_acc" in history_df.columns and history_df["val_acc"].notna().any():
        ax2.plot(epochs, history_df["val_acc"] * 100.0, label="Valid Acc (%)")
    ax2.set_title(f"{title_label} - Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_single_fold(model_name, cfg, fold, epochs_override, device):
    ModelClass = cfg["model"]
    model_args = cfg["model_args"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    epochs = epochs_override if epochs_override is not None else cfg["epoch"]
    lr = cfg["lr"]

    print(f"\n=== [{model_name}] Fold {fold} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} ===")

    train_dataset = getImageLabel(fold=fold, split="Train", img_size=img_size)
    valid_dataset = getImageLabel(fold=fold, split="Valid", img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = ModelClass(**model_args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("Checkpoints", exist_ok=True)
    results_dir = os.path.join("Results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    best_val_acc = -1.0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.permute(0, 3, 1, 2).to(device)
            _, label_indices = torch.max(labels, 1)
            label_indices = label_indices.to(device)

            outputs = model(images)
            loss = criterion(outputs, label_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += label_indices.size(0)
            correct += predicted.eq(label_indices).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate_loader(model, valid_loader, criterion, device)

        history.append(
            {
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"[{model_name} | Fold {fold}] Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | "
            f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join("Checkpoints", f"best_{model_name}_fold{fold}.pth")
            torch.save(model.state_dict(), ckpt_path)

    history_df = pd.DataFrame(history)
    csv_path = os.path.join(results_dir, f"training_history_fold{fold}.csv")
    plot_path = os.path.join(results_dir, f"training_curve_fold{fold}.png")

    history_df.to_csv(csv_path, index=False)
    plot_training_curves(history_df, f"{model_name} (Fold {fold})", plot_path)
    print(f"--> Tersimpan: {csv_path} & {plot_path}")


def train_model():
    parser = argparse.ArgumentParser(description="Training Script Multi-Fold untuk Model Klasifikasi")
    parser.add_argument(
        "--model",
        type=str,
        default="model1",
        choices=list(CONFIGS.keys()),
        help="Pilih model dari config.py",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Daftar fold yang akan dilatih, contoh: --folds 1 2 3 4 5",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override jumlah epoch dari config.py (opsional)",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat: {device} | Model: {args.model} | Folds: {args.folds}")

    for fold in args.folds:
        run_single_fold(args.model, cfg, fold, args.epochs, device)

    print("\nSelesai semua fold!")


if __name__ == "__main__":
    train_model()