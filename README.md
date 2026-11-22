# Klasifikasi Lesi Kulit Mpox (MSLD v2.0)

Klasifikasi citra lesi kulit ke dalam **6 kelas** menggunakan model deep learning berbasis PyTorch dan YOLO (Ultralytics). Evaluasi dilakukan dengan **5-fold cross validation**.

Kelas: `Chickenpox`, `Cowpox`, `HFMD`, `Healthy`, `Measles`, `Monkeypox`

---

## Daftar Isi

- [Dataset](#dataset)
- [Model](#model)
- [Struktur Folder](#struktur-folder)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Output](#output)
- [Tim](#tim)

---

## Dataset

Dataset: [Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20)

Dataset **diunduh otomatis** lewat `kagglehub` saat script pertama kali dijalankan (selanjutnya memakai cache), jadi tidak perlu download manual.

Skema pembagian data (per fold, fold 1 sampai 5):

| Split | Sumber |
|-------|--------|
| Train | `Original Images` + `Augmented Images` (salinan `_ORIGINAL` di folder augmented dilewati agar tidak dobel) |
| Valid | `Original Images` |
| Test  | `Original Images` |

Gambar dibaca dengan OpenCV, diubah ke RGB, di-resize, lalu dinormalisasi ke rentang 0-1.

## Model

| Nama | Arsitektur | Ukuran Gambar | Batch | Epoch | Learning Rate |
|------|-----------|---------------|-------|-------|---------------|
| `model1` | SimpleCNN | 100x100 | 32 | 100 | 0.001 |
| `model2` | SimpleCNN2 (2 blok Conv-ReLU-MaxPool + 2 layer Linear) | 100x100 | 16 | 50 | 0.0001 |
| `yolo` | YOLO11n-cls (`yolo11n-cls.yaml`, from scratch / tanpa pretrained) | 224x224 | default | 35 (patience 10) | default Ultralytics |

Semua pengaturan model CNN ada di `config.py` (dictionary `CONFIGS`). Optimizer: Adam, loss: CrossEntropyLoss.

## Struktur Folder

```
mpox-skin-lesion-classification/
├── Checkpoints/          # bobot model terbaik (tidak di-upload ke GitHub)
├── Models/
│   ├── cnn1.py           # SimpleCNN
│   └── cnn2.py           # SimpleCNN2
├── Results/              # hasil training & testing per model
├── Utils/
│   └── getData.py        # dataset loader (kagglehub + OpenCV)
├── config.py             # CONFIGS & YOLO_CONFIGS
├── train.py              # training CNN (multi-fold)
├── yolo.py               # training YOLO klasifikasi
├── test.py               # evaluasi model1 / model2 / yolo
├── requirements.txt
└── README.md
```

## Instalasi

```bash
git clone https://github.com/nurakhadiaramadhani-netizen/mpox-skin-lesion-classification.git
cd mpox-skin-lesion-classification

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Library utama: `torch`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`, `ultralytics`.

> Disarankan memakai GPU (CUDA). Script otomatis memakai CPU jika GPU tidak tersedia.

## Cara Penggunaan

Jalankan semua perintah dari **folder utama proyek** (yang berisi `train.py`).

### 1. Training model CNN

```bash
# model1, semua fold (default)
python train.py --model model1

# model2, hanya fold tertentu
python train.py --model model2 --folds 1 2 3

# override jumlah epoch dari config.py
python train.py --model model1 --epochs 20
```

| Argumen | Keterangan | Default |
|---------|-----------|---------|
| `--model` | `model1` atau `model2` | `model1` |
| `--folds` | daftar fold yang dilatih | `1 2 3 4 5` |
| `--epochs` | override epoch dari `config.py` | sesuai config |

Checkpoint terbaik (berdasarkan akurasi validasi) disimpan per fold di `Checkpoints/best_<model>_fold<N>.pth`.

### 2. Training YOLO

```bash
python yolo.py
```

Saat ini YOLO dilatih pada **fold 1** saja. Hasilnya disimpan oleh Ultralytics di `runs/classify/`.

### 3. Testing / evaluasi

```bash
python test.py --model model1
python test.py --model model2
python test.py --model yolo --folds 1

# opsi tambahan
python test.py --model model1 --folds 5 --split Test
```

| Argumen | Keterangan | Default |
|---------|-----------|---------|
| `--model` | `model1`, `model2`, atau `yolo` | wajib |
| `--folds` | jumlah fold yang dievaluasi | `5` |
| `--split` | `Train`, `Valid`, atau `Test` | `Test` |

## Output

Hasil training (`Results/<model>/`):
- `training_history_fold<N>.csv` : loss & akurasi tiap epoch
- `training_curve_fold<N>.png` : grafik loss & akurasi

Hasil testing (`Results/<model>/`):
- `confusion_matrix_fold<N>.png` dan `confusion_matrix_all.png`
- `classification_report_fold<N>.txt`
- `test_metrics.csv` : accuracy, precision, recall, F1 (macro), dan ROC-AUC (OvR) per fold, beserta mean dan std

## Tim

- Abdul Aziz
- Nadya Rahmawati
- Nur Akhadia Ramadhani

## Disclaimer

Proyek ini dibuat untuk keperluan akademik dan **bukan pengganti diagnosis medis profesional**.