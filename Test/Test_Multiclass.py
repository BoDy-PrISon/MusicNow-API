import os
import pickle
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from Classes.CRNN_class import CRNN
from torch.utils.data import DataLoader

# Пути к моделям и данным
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# Загрузка энкодера
with open(os.path.join(MODELS_DIR, 'genre_label_encoder_multiclass.pkl'), 'rb') as f:
    le = pickle.load(f)

# Загрузка обучающей выборки (пусть путь к данным будет ../fma_metadata/tracks.csv)
import pandas as pd
tracks = pd.read_csv(os.path.join(BASE_DIR, 'fma_metadata', 'tracks.csv'), header=[0, 1], index_col=0)

files = []
labels = []
for idx, row in tracks.iterrows():
    tid = '{:06d}'.format(idx)
    path = os.path.join('E:/fma_large', tid[:3], tid + '.mp3')
    if not os.path.exists(path):
        continue
    genre = row['track', 'genre_top']
    if pd.isna(genre):
        continue
    files.append(path)
    labels.append(genre)

# Кодирование жанров
labels_encoded = le.transform(labels)

# 33% для теста
from sklearn.model_selection import train_test_split
files = np.array(files)
labels_encoded = np.array(labels_encoded)
_, test_files, _, test_labels = train_test_split(files, labels_encoded, test_size=0.33, random_state=42, stratify=labels_encoded)

# DataLoader
from torch.utils.data import Dataset
class FMAMultiClassDataset(Dataset):
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
        import librosa
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
BATCH_SIZE = 8
test_dataset = FMAMultiClassDataset(test_files, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Загрузка модели
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(n_mels=128, n_classes=len(le.classes_)).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'crnn_fma_multiclass.pth'), map_location=DEVICE))
model.eval()

# Тестирование
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb)
        pred_labels = preds.argmax(1).cpu().numpy()
        all_preds.extend(pred_labels)
        all_true.extend(yb.numpy())
all_preds = np.array(all_preds)
all_true = np.array(all_true)

print("Accuracy:", accuracy_score(all_true, all_preds))
print(classification_report(all_true, all_preds, target_names=le.classes_))

# Матрица ошибок
cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# График точности по жанрам
from sklearn.metrics import precision_recall_fscore_support
_, _, f1s, _ = precision_recall_fscore_support(all_true, all_preds, average=None)
plt.figure(figsize=(10, 5))
plt.bar(le.classes_, f1s)
plt.xticks(rotation=90)
plt.ylabel('F1-score')
plt.title('F1-score по жанрам')
plt.tight_layout()
plt.show() 