# Klasifikasi Lesi Kulit Mpox

Proyek klasifikasi citra lesi kulit **Mpox** dan kondisi kulit lainnya menggunakan skema **5-Fold Cross-Validation**. Seluruh model pada eksperimen ini—meliputi dua arsitektur CNN kustom (`model1` dan `model2`) serta model klasifikasi `YOLO11n-cls` (`yolo11n-cls.yaml` menggunakan *library* Ultralytics)—dilatih **dari nol (*from scratch*)**.

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
- [Artefak & Hasil Evaluasi](#artefak--hasil-evaluasi)
- [Catatan Eksperimen](#catatan-eksperimen)

## Gambaran Umum

- **CNN *From Scratch* (`model1` & `model2`)**: Arsitektur CNN yang didefinisikan dan dilatih langsung dari awal pada masing-masing *fold*.
- **YOLO11 Classification *From Scratch* (`yolo11n-cls.yaml`)**: Model klasifikasi Ultralytics yang diinisialisasi dari arsitektur `yolo11n-cls.yaml` (tanpa bobot *pretrained*) dan dilatih dari awal.
- **Evaluasi Multi-Fold**: Setiap model diuji pada skema *5-fold* (`fold1` s.d. `fold5`) dengan pencatatan riwayat per *epoch*, kurva *loss* & akurasi, *confusion matrix*, serta *classification report*.

## Struktur Proyek

```text
mpox-skin-lesion-classification/
├── Checkpoints/          # Penyimpanan bobot model (checkpoint) hasil pelatihan
├── Models/               # Definisi arsitektur jaringan CNN (cnn1.py, cnn2.py, dll.)
├── Utils/                # Utilitas pemuatan dataset (getData.py), pembagian fold, metrik, & visualisasi
├── Results/              # Hasil eksperimen terstandardisasi per model
│   ├── model1/           # training_history_fold*.csv & training_curve_fold*.png
│   ├── model2/           # training_history_fold*.csv & training_curve_fold*.png
│   └── yolo/             # classification_report_fold*.txt, confusion_matrix_fold*.png,
│                         # confusion_matrix_all.png, test_metrics.csv,
│                         # training_history_fold*.csv, & training_curve_fold*.png
├── runs/                 # Log dan keluaran bawaan Ultralytics YOLO (runs/classify/train*)
│   └── classify/
├── config.py             # Konfigurasi path, hyperparameter, & pengaturan eksperimen
├── train.py              # Skrip pelatihan multi-fold untuk model1 dan model2
├── yolo.py               # Skrip pelatihan klasifikasi YOLO from scratch (yolo11n-cls.yaml)
├── test.py               # Skrip pengujian model & pembuatan laporan metrik evaluasi
├── yolo11n-cls.pt        # Berkas bobot YOLO11 di direktori proyek (yolo.py memakai yolo11n-cls.yaml)
└── README.md             # Dokumentasi proyek