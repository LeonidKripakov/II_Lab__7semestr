import os
import glob
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ----------------------------
# Конфиг
# ----------------------------
@dataclass
class Config:
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 320  # ~20ms при 16kHz
    win_length: int = 1024
    max_seconds: float = 3.5   # обрезка/паддинг до фиксированной длины
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 30
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Маппинг эмоций под RAVDESS
# ----------------------------
RAVDESS_EMOTION_ID_TO_NAME = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

# Можно убрать "calm" или объединять классы — это уже твоя задача.
USED_EMOTIONS = [1, 2, 3, 4, 5, 6, 7, 8]


def parse_ravdess_emotion_id(filepath: str) -> int:
    """
    RAVDESS: '03-01-05-01-02-02-12.wav' -> emotion_id = 5
    """
    name = os.path.basename(filepath)
    parts = name.replace(".wav", "").split("-")
    if len(parts) < 3:
        raise ValueError(f"Не похоже на RAVDESS имя: {name}")
    emotion_id = int(parts[2])
    return emotion_id


# ----------------------------
# Dataset
# ----------------------------
class SpeechEmotionDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], cfg: Config):
        self.files = files
        self.labels = labels
        self.cfg = cfg

        self.resampler = None  # создадим при необходимости
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            melkwargs={
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "win_length": cfg.win_length,
                "n_mels": 64,
                "center": True,
            },
        )

    def _load_audio(self, path: str) -> torch.Tensor:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True)   # [time, channels]
        wav = torch.from_numpy(data.T).float()     # -> [channels, time]
        wav = wav.mean(dim=0, keepdim=True)        # mono: [1, time]

        if sr != self.cfg.sample_rate:
            self.resampler = self.resampler or torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
            wav = self.resampler(wav)

        wav = wav / (wav.abs().max() + 1e-9)
        return wav


    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        max_len = int(self.cfg.sample_rate * self.cfg.max_seconds)
        cur_len = wav.shape[-1]
        if cur_len > max_len:
            wav = wav[..., :max_len]
        elif cur_len < max_len:
            pad = max_len - cur_len
            wav = F.pad(wav, (0, pad))
        return wav

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]
        label = self.labels[idx]

        wav = self._load_audio(path)
        wav = self._fix_length(wav)

        # MFCC: [1, n_mfcc, time_frames]
        feat = self.mfcc(wav)

        # Лёгкая стабилизация
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)

        return feat, label


# ----------------------------
# Модель: CNN -> BiGRU -> классификатор
# ----------------------------
class CNNBiGRU(nn.Module):
    def __init__(self, n_classes: int, n_mfcc: int = 40):
        super().__init__()
        # вход: [B, 1, n_mfcc, T]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # mfcc/2, T/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # mfcc/4, T/4
        )

        # После CNN: [B, C, mfcc', T']
        # Превращаем в последовательность по времени T', фичи = C*mfcc'
        self.gru_hidden = 128
        self.gru = nn.GRU(
            input_size=64 * (n_mfcc // 4),
            hidden_size=self.gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(self.gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mfcc, T]
        z = self.cnn(x)  # [B, 64, mfcc', T']
        b, c, mf, t = z.shape
        z = z.permute(0, 3, 1, 2).contiguous()  # [B, T', C, mfcc']
        z = z.view(b, t, c * mf)               # [B, T', C*mfcc']
        z, _ = self.gru(z)                     # [B, T', 2H]
        z = z.mean(dim=1)                      # глобальный pooling по времени
        logits = self.head(z)                  # [B, n_classes]
        return logits


# ----------------------------
# Train/Eval
# ----------------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)  # [B, 1, n_mfcc, T]
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


def build_file_list_ravdess(data_root: str) -> Tuple[List[str], List[int], Dict[int, int], List[str]]:
    wavs = glob.glob(os.path.join(data_root, "**", "*.wav"), recursive=True)
    files, labels_raw = [], []
    for p in wavs:
        try:
            eid = parse_ravdess_emotion_id(p)
        except Exception:
            continue
        if eid in USED_EMOTIONS:
            files.append(p)
            labels_raw.append(eid)

    # Перенумеруем классы в 0..K-1
    unique = sorted(set(labels_raw))
    id_to_class = {eid: i for i, eid in enumerate(unique)}
    labels = [id_to_class[eid] for eid in labels_raw]
    class_names = [RAVDESS_EMOTION_ID_TO_NAME[eid] for eid in unique]
    return files, labels, id_to_class, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Папка с wav (например data/RAVDESS)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_seconds", type=float, default=3.5)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_seconds=args.max_seconds)
    print("Device:", cfg.device)

    files, labels, id_to_class, class_names = build_file_list_ravdess(args.data_root)
    if len(files) == 0:
        raise RuntimeError("Не нашёл .wav. Проверь --data_root и структуру папок.")

    # stratify чтобы классы были похожи в train/val
    X_train, X_val, y_train, y_val = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = SpeechEmotionDataset(X_train, y_train, cfg)
    val_ds = SpeechEmotionDataset(X_val, y_val, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = CNNBiGRU(n_classes=len(class_names), n_mfcc=cfg.n_mfcc).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print("Classes:", class_names)
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, cfg.device)
        va_loss, va_acc, y_true, y_pred = eval_one_epoch(model, val_loader, cfg.device)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = os.path.join("checkpoints", "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"Saved: {ckpt_path} (val acc {best_val_acc:.3f})")

    print("\nClassification report (val):")
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    main()
