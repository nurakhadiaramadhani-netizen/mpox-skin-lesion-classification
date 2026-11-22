import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from Utils.getData import CLASSES, getImageLabel, get_dataset_root
from config import CONFIGS, YOLO_CONFIGS

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Ukuran gambar YOLO harus sama dengan saat training (yolo.py memakai imgsz=224)
YOLO_IMG_SIZE = 224


def plot_confusion_matrix(cf_matrix, classes, title, path):
    row_sum = np.sum(cf_matrix, axis=1)[:, None]
    normalized = np.divide(
        cf_matrix,
        row_sum,
        out=np.zeros_like(cf_matrix, dtype=float),
        where=row_sum != 0,
    )
    df_cm = pd.DataFrame(normalized, index=classes, columns=classes)

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Prediksi")
    plt.ylabel("Label asli")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def to_probs(raw_out: torch.Tensor) -> torch.Tensor:
    if raw_out.max().item() <= 1.0001 and raw_out.min().item() >= -1e-6:
        if torch.allclose(
            raw_out.sum(dim=1),
            torch.ones(raw_out.size(0), device=raw_out.device),
            atol=1e-2,
        ):
            return raw_out.clamp(0.0, 1.0)
    if raw_out.max().item() <= 1e-4:
        return torch.exp(raw_out)
    return F.softmax(raw_out, dim=1)


def compute_fold_metrics(y_true, y_pred, y_probs):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def save_fold_artifacts(model_dir, fold, y_true, y_pred, y_probs, classes):
    os.makedirs(model_dir, exist_ok=True)
    cf = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plot_confusion_matrix(
        cf,
        classes,
        title=f"Confusion Matrix - Fold {fold}",
        path=os.path.join(model_dir, f"confusion_matrix_fold{fold}.png"),
    )

    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    with open(
        os.path.join(model_dir, f"classification_report_fold{fold}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_txt)

    return cf, compute_fold_metrics(y_true, y_pred, y_probs)


def save_summary_csv_and_all_cm(model_dir, fold_rows, all_y_true, all_y_pred, classes):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.DataFrame(fold_rows)
    numeric_cols = [c for c in METRICS if c in df.columns]

    summary_rows = []
    for stat_name, fn in [("mean", np.nanmean), ("std", np.nanstd)]:
        row = {"fold": stat_name}
        for col in numeric_cols:
            row[col] = fn(df[col].to_numpy(dtype=float))
        summary_rows.append(row)

    out_df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    out_df.to_csv(os.path.join(model_dir, "test_metrics.csv"), index=False)

    cf_all = confusion_matrix(
        all_y_true, all_y_pred, labels=list(range(len(classes)))
    )
    plot_confusion_matrix(
        cf_all,
        classes,
        title="Confusion Matrix - All Folds",
        path=os.path.join(model_dir, "confusion_matrix_all.png"),
    )


def predict_cnn_fold(model_name, fold, split, device, root):
    cfg = CONFIGS[model_name]
    model_cls = cfg["model"]
    model_args = cfg.get("model_args", {})
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]

    # Checkpoint per fold, sama dengan nama yang disimpan train.py
    ckpt_path = os.path.join("Checkpoints", f"best_{model_name}_fold{fold}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {ckpt_path}. "
            f"Jalankan: python train.py --model {model_name} --folds {fold}"
        )
    print(f"[{model_name}] Fold {fold} memakai checkpoint: {ckpt_path}")

    dataset = getImageLabel(fold=fold, split=split, img_size=img_size, root=root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model_cls(**model_args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    y_true_list, y_pred_list, y_probs_list = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.permute(0, 3, 1, 2).to(device)
            targets = torch.argmax(labels, dim=1).numpy()

            outputs = model(images)
            probs = to_probs(outputs).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true_list.append(targets)
            y_pred_list.append(preds)
            y_probs_list.append(probs)

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_probs_list),
    )


def _find_yolo_weights(fold: int) -> str:
    # Cari bobot per-fold jika ada (misal fold1/weights/best.pt atau *fold*1*...)
    fold_patterns = [
        os.path.join("runs", "classify", f"*fold*{fold}*", "weights", "best.pt"),
        os.path.join("runs", "classify", f"*fold{fold}*", "weights", "best.pt"),
        os.path.join("runs", "classify", f"train*fold*{fold}*", "weights", "best.pt"),
    ]
    for pat in fold_patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[-1]

    # Jika tidak ada bobot per-fold khusus, pakai best.pt terbaru di runs/classify/
    all_bests = sorted(glob.glob(os.path.join("runs", "classify", "*", "weights", "best.pt")))
    if all_bests:
        return all_bests[-1]

    yolo_cfg = next(iter(YOLO_CONFIGS.values()), {})
    fallback = yolo_cfg.get("model_name", "yolo11n-cls.pt")
    if os.path.exists(fallback):
        return fallback
    if os.path.exists("yolo11n-cls.pt"):
        return "yolo11n-cls.pt"
    return fallback


def _align_yolo_probs(raw_probs: np.ndarray, model_names_dict: dict) -> np.ndarray:
    num_classes = len(CLASSES)
    aligned = np.zeros((raw_probs.shape[0], num_classes), dtype=float)
    name_to_idx = {str(v).strip().lower(): int(k) for k, v in model_names_dict.items()}

    for target_idx, cls_name in enumerate(CLASSES):
        src_idx = name_to_idx.get(cls_name.lower())
        if src_idx is not None and src_idx < raw_probs.shape[1]:
            aligned[:, target_idx] = raw_probs[:, src_idx]
        elif target_idx < raw_probs.shape[1]:
            aligned[:, target_idx] = raw_probs[:, target_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(
        aligned,
        row_sums,
        out=np.full_like(aligned, 1.0 / num_classes),
        where=row_sums > 0,
    )
    return aligned


def predict_yolo_fold(fold, split, device, root):
    from ultralytics import YOLO

    yolo_cfg = next(iter(YOLO_CONFIGS.values()), {})
    img_size = YOLO_IMG_SIZE
    batch_size = yolo_cfg.get("batch_size", 16)
    weights_path = _find_yolo_weights(fold)
    print(f"[yolo] Fold {fold} memakai bobot: {weights_path}")

    dataset = getImageLabel(fold=fold, split=split, img_size=img_size, root=root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    yolo_model = YOLO(weights_path)
    model_names = getattr(yolo_model, "names", {}) or {}

    y_true_list, y_pred_list, y_probs_list = [], [], []
    for images, labels in loader:
        targets = torch.argmax(labels, dim=1).numpy()
        # images: (B, H, W, 3) float [0..1] -> uint8 [0..255] numpy list
        imgs_np = [(img.numpy() * 255.0).clip(0, 255).astype(np.uint8) for img in images]

        results = yolo_model.predict(
            source=imgs_np,
            imgsz=img_size,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )

        batch_probs = []
        for res in results:
            if res.probs is not None:
                p = res.probs.data.detach().cpu().numpy()
            else:
                p = np.zeros(len(CLASSES), dtype=float)
            batch_probs.append(p)

        raw_probs = np.vstack(batch_probs)
        aligned_probs = _align_yolo_probs(raw_probs, model_names)
        preds = np.argmax(aligned_probs, axis=1)

        y_true_list.append(targets)
        y_pred_list.append(preds)
        y_probs_list.append(aligned_probs)

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_probs_list),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["model1", "model2", "yolo"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split", type=str, default="Test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_dataset_root()
    model_dir = os.path.join("Results", args.model)

    fold_rows = []
    all_y_true, all_y_pred = [], []

    for fold in range(1, args.folds + 1):
        if args.model in ("model1", "model2"):
            y_true, y_pred, y_probs = predict_cnn_fold(
                args.model, fold, args.split, device, root
            )
        else:
            y_true, y_pred, y_probs = predict_yolo_fold(fold, args.split, device, root)

        _, fold_metrics = save_fold_artifacts(
            model_dir, fold, y_true, y_pred, y_probs, CLASSES
        )
        row = {"fold": fold, **fold_metrics}
        fold_rows.append(row)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        print(f"[{args.model}] Fold {fold} selesai | acc={fold_metrics['accuracy']:.4f}")

    save_summary_csv_and_all_cm(
        model_dir,
        fold_rows,
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred),
        CLASSES,
    )
    print(f"Semua hasil tersimpan di: {model_dir}")


if __name__ == "__main__":
    main()