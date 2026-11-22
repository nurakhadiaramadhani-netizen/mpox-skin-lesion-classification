import os
import cv2 as cv
import numpy as np
import torch
import kagglehub
from torch.utils.data import Dataset

# Urutan kelas (alfabetis, sama dengan urutan folder). Indeks di list ini = label kelas.
CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']


def get_dataset_root():
    # Unduh dataset dari Kaggle (hanya sekali, berikutnya memakai cache)
    return kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")


class getImageLabel(Dataset):
    def __init__(self, fold=1, split='Train', img_size=100, root=None):
        # fold     : nomor fold, 1 sampai 5
        # split    : 'Train', 'Valid', atau 'Test'
        # img_size : ukuran gambar setelah di-resize (100 untuk SimpleCNN, 224 untuk model pretrained)
        # root     : lokasi dataset; kalau kosong, diambil lewat kagglehub
        if root is None:
            root = get_dataset_root()

        self.dataset = []

        # initiate the one hot encoding for class
        to_one_hot = np.eye(len(CLASSES))

        # Valid dan Test hanya dari Original Images.
        # Train = Train asli + Train augmented.
        folders = [os.path.join(root, 'Original Images', 'Original Images', 'FOLDS', f'fold{fold}', split)]
        if split == 'Train':
            folders.append(os.path.join(root, 'Augmented Images', 'Augmented Images', 'FOLDS_AUG', f'fold{fold}_AUG', 'Train'))

        for folder in folders:
            for i, class_name in enumerate(CLASSES):
                class_folder = os.path.join(folder, class_name)
                for image_name in sorted(os.listdir(class_folder)):
                    # folder augmented sudah memuat salinan gambar asli (_ORIGINAL), lewati supaya tidak dobel
                    if '_ORIGINAL' in image_name:
                        continue

                    image = cv.imread(os.path.join(class_folder, image_name))
                    if image is None:  # file rusak / bukan gambar
                        continue

                    # OpenCV membaca BGR, ubah ke RGB, lalu resize
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image = cv.resize(image, (img_size, img_size))

                    # disimpan sebagai uint8 (0-255) supaya hemat memori, dinormalisasi di __getitem__
                    self.dataset.append([image, to_one_hot[i]])

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        # normalisasi ke 0-1, bentuk gambar (tinggi, lebar, 3)
        return torch.tensor(feature / 255, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # Uji cepat: python Utils/getData.py
    root = get_dataset_root()
    print("Lokasi dataset:", root)

    for split in ['Train', 'Valid', 'Test']:
        data = getImageLabel(fold=1, split=split, img_size=100, root=root)
        labels = [int(np.argmax(label)) for _, label in data.dataset]
        per_class = {c: labels.count(i) for i, c in enumerate(CLASSES)}
        print(f"\nFold 1 - {split}: {len(data)} gambar")
        print("  per kelas:", per_class)

    feature, label = data[0]
    print("\nContoh satu data (Test):")
    print("  bentuk gambar :", tuple(feature.shape), "| min-max:", round(feature.min().item(), 3), "-", round(feature.max().item(), 3))
    print("  label one-hot :", label.tolist(), "->", CLASSES[int(label.argmax())])
