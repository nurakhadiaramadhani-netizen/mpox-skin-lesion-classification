# ==========================================
# FILE: config.py
# ==========================================
# Daftar semua model beserta pengaturannya.
# Untuk menambah model: buat file model di Models/, impor di sini, lalu tambahkan satu entri baru.
#
# Keterangan setiap kunci:
#   model      : class model (harus menerima input (B, 3, H, W) dan mengeluarkan log-probabilitas 6 kelas)
#   model_args : argumen yang diberikan saat membuat model
#   img_size   : ukuran gambar setelah di-resize
#   batch_size : jumlah gambar per batch
#   epoch      : jumlah epoch per fold
#   lr         : learning rate untuk optimizer Adam
#   patience   : early stopping; berhenti jika valid loss tidak membaik selama N epoch (None = tidak dipakai)

from Models.cnn1 import SimpleCNN
from Models.cnn2 import SimpleCNN2  # Impor model kedua dari cnn2.py

CONFIGS = {
    # Model 1: SimpleCNN standar
    "model1": {
        "model": SimpleCNN,
        "model_args": {},
        "img_size": 100,
        "batch_size": 32,
        "epoch": 100,
        "lr": 0.001,
        "patience": None,
    },

    # Model 2: SimpleCNN2 dengan variasi parameter (batch_size & lr diubah, epoch disamakan jadi 100)
    "model2": {
        "model": SimpleCNN2,  
        "model_args": {},
        "img_size": 100,
        "batch_size": 16,
        "epoch": 50,
        "lr": 0.0001,
        "patience": None,
    }
}

# Tambahkan konfigurasi ini di dalam file config.py kamu

YOLO_CONFIGS = {
    "yolo_v8n": {
        "model_name": "yolov8n.pt",  # Bisa diganti path bobot custom nanti (.pt)
        "img_size": 640,
        "epochs": 50,
        "batch_size": 16,
        "conf_threshold": 0.5,
    }
}