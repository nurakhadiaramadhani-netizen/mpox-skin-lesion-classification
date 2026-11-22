import os
from ultralytics import YOLO
import torch
from Utils.getData import get_dataset_root, CLASSES

def train_yolo():
    print("🚀 Memulai Inisialisasi Training YOLO Klasifikasi (From Scratch)...")

    # 1. Deteksi perangkat (CPU atau GPU CUDA)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Perangkat target: {DEVICE}")

    # 2. Ambil root dataset secara otomatis dari kagglehub (sama seperti getData.py)
    root = get_dataset_root()
    print(f"Lokasi dataset cache Kaggle: {root}")

    # 3. Tentukan path folder Train dan Validasi dari struktur Kaggle
    train_path = os.path.join(
        root, 'Augmented Images', 'Augmented Images', 'FOLDS_AUG', 'fold1_AUG', 'Train'
    )
    val_path = os.path.join(
        root, 'Original Images', 'Original Images', 'FOLDS', 'fold1', 'Valid'
    )

    print(f"📂 Folder Train YOLO: {train_path}")
    print(f"📂 Folder Validasi YOLO: {val_path}")
    print(f"Jumlah kelas: {len(CLASSES)}")
    print(f"Nama kelas: {CLASSES}"   )

    # 4. Muat arsitektur YOLO klasifikasi dari .yaml (tanpa bobot pretrained / from scratch)
    model = YOLO("yolo11n-cls.yaml")
    
    # Tampilkan informasi ringkas model
    model.info()

    # 5. Jalankan proses training otomatis oleh Ultralytics
    print("\n⏳ Proses training YOLO (from scratch) sedang berjalan...")
    results = model.train(
        data=train_path,  # Mengarah langsung ke folder train dataset Kaggle
        epochs=35,        # Jumlah epoch
        imgsz=224,        # Ukuran standar gambar untuk model YOLO cls
        device=DEVICE,     # Otomatis menyesuaikan GPU atau CPU
        patience=10,
    )

    print("✅ Training YOLO Selesai!")

if __name__ == "__main__":
    train_yolo()