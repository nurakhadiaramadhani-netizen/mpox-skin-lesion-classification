import os
import pandas as pd

names = {"model1": "model1 (SimpleCNN)", "model2": "model2 (SimpleCNN2)", "yolo": "yolo (YOLO11n-cls)"}
cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
         "|-------|----------|-----------|--------|----|---------|"]

for key, label in names.items():
    path = os.path.join("Results", key, "test_metrics.csv")
    if not os.path.exists(path):
        lines.append(f"| {label} | - | - | - | - | - |")
        continue
    df = pd.read_csv(path)
    df["fold"] = df["fold"].astype(str)
    fold_rows = df[~df["fold"].isin(["mean", "std"])]
    if len(fold_rows) == 1:
        r = fold_rows.iloc[0]
        vals = " | ".join(f"{r[c]:.3f}" for c in cols)
        lines.append(f"| {label} (fold {r['fold']} saja) | {vals} |")
    else:
        mean = df[df["fold"] == "mean"].iloc[0]
        std = df[df["fold"] == "std"].iloc[0]
        vals = " | ".join(f"{mean[c]:.3f} ± {std[c]:.3f}" for c in cols)
        lines.append(f"| {label} | {vals} |")

text = "\n".join(lines)
print(text)
open("tabel_hasil.md", "w", encoding="utf-8").write(text + "\n")
