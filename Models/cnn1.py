import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, input_dim=100, input_c=3, output=6, hidden_dim=128, dropout=0.5, device='cpu'):
        # Arsitektur sama persis dengan materi; hanya output diubah menjadi 6 kelas.
        # Catatan: input_dim, hidden_dim, dropout, dan device ada di materi tetapi tidak dipakai di arsitektur.
        # Ukuran lapisan Linear (50 * 9 * 9) hanya cocok untuk gambar 100 x 100.
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        self.fc = nn.Linear(in_features=50 * 9 * 9, out_features=output)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)   # (B, 3, 100, 100) -> (B, 20, 32, 32)
        x = self.convBlock2(x)   # (B, 20, 32, 32)  -> (B, 50, 9, 9)

        x = self.fc(torch.flatten(x, 1))   # (B, 50*9*9) -> (B, 6)
        x = self.logSoftmax(x)             # log-probabilitas 6 kelas

        return x


if __name__ == "__main__":
    # Uji cepat: python Models/CNN.py
    model = SimpleCNN()
    dummy = torch.rand(1, 3, 100, 100)   # 1 gambar acak, 3 channel, 100 x 100
    out = model(dummy)
    print("Bentuk input  :", tuple(dummy.shape))
    print("Bentuk output :", tuple(out.shape))           # harus (1, 6)
    print("Jumlah peluang:", round(out.exp().sum().item(), 4))  # harus 1.0
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))
