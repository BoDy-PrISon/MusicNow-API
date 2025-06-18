from torch.utils.data import Dataset
import numpy as np
import librosa
import torch


class FMADataset(Dataset):
    def __init__(self, files, labels, n_mels=128, duration=30, sr=22050):
        self.files = files
        self.labels = labels
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.samples = sr * duration

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        try:
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
        except Exception as e:
            print(f'Ошибка при загрузке {file_path}: {e}')
            return self.__getitem__((idx + 1) % len(self.files))
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_db = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        return mel_db, label


class FMAMultiLabelDataset(Dataset):
    def __init__(self, files, labels, n_mels=128, duration=30, sr=22050):
        self.files = files
        self.labels = labels
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.samples = sr * duration

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        try:
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
        except Exception as e:
            print(f'Ошибка при загрузке {file_path}: {e}')
            return self.__getitem__((idx + 1) % len(self.files))
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_db = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)
        return mel_db, label