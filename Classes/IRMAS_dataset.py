import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from pathlib import Path

# 11 инструментов IRMAS
IRMAS_INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']


class IRMASDataset(Dataset):
    def __init__(self, data_dir, n_mels=128, duration=3, sr=22050, train=True):
        self.data_dir = Path(data_dir)
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.samples = sr * duration

        # Подготавливаем список файлов и меток
        self.file_list = []
        self.labels = []

        self._prepare_data()

        # Создаём MultiLabelBinarizer для преобразования меток
        self.mlb = MultiLabelBinarizer(classes=IRMAS_INSTRUMENTS)
        self.labels_encoded = self.mlb.fit_transform(self.labels)

        print(f"Загружено {len(self.file_list)} файлов")
        print(f"Распределение по инструментам:")
        for i, instrument in enumerate(IRMAS_INSTRUMENTS):
            count = sum(1 for label in self.labels if instrument in label)
            print(f"  {instrument}: {count} файлов")

    def _prepare_data(self):
        """Подготавливает список файлов и соответствующих меток"""
        for instrument_dir in self.data_dir.iterdir():
            if instrument_dir.is_dir() and instrument_dir.name in IRMAS_INSTRUMENTS:
                instrument = instrument_dir.name

                # Проходим по всем WAV файлам в папке инструмента
                for wav_file in instrument_dir.glob("*.wav"):
                    self.file_list.append(str(wav_file))
                    # Для IRMAS каждый файл содержит только один инструмент
                    self.labels.append([instrument])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels_encoded[idx]

        try:
            # Загружаем аудио
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)

            # Дополняем или обрезаем до нужной длины
            if len(y) < self.samples:
                y = np.pad(y, (0, self.samples - len(y)))
            else:
                y = y[:self.samples]

            # Создаём мел-спектрограмму
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Нормализация
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

            # Преобразуем в тензор
            mel_db = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # Добавляем канал
            label = torch.tensor(label, dtype=torch.float32)

            return mel_db, label

        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
            # Возвращаем следующий файл в случае ошибки
            return self.__getitem__((idx + 1) % len(self.file_list))