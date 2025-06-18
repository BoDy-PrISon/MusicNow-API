import os
import pickle
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from Classes.CRNN_class import CRNN, FMAMultiLabelDataset
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Пути к моделям и данным
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# Загрузка энкодера и жанров
with open(os.path.join(MODELS_DIR, 'genre_label_encoder_multiLabel.pkl'), 'rb') as f:
    le = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'all_genres_multiLabel.pkl'), 'rb') as f:
    all_genres = pickle.load(f)

# Загрузка обучающей выборки (пусть путь к данным будет ../fma_metadata/tracks.csv)

tracks = pd.read_csv(os.path.join(BASE_DIR, 'fma_metadata', 'tracks.csv'), header=[0, 1], index_col=0)
genres_df = pd.read_csv(os.path.join(BASE_DIR, 'fma_metadata', 'genres.csv'), index_col=0)
genre_id_to_name = dict(zip(genres_df.index, genres_df['title']))

files = []
labels = []
for idx, row in tracks.iterrows():
    tid = '{:06d}'.format(idx)
    path = os.path.join('E:/fma_large', tid[:3], tid + '.mp3')
    if not os.path.exists(path):
        continue
    genre_ids = row['track', 'genres']
    if pd.isna(genre_ids) or genre_ids == '[]':
        continue
    import json
    genre_ids = json.loads(genre_ids.replace("'", '"'))
    genre_names = [genre_id_to_name[g] for g in genre_ids if g in genre_id_to_name]
    if not genre_names:
        continue
    files.append(path)
    labels.append(genre_names)

# Multi-hot encoding
n_classes = len(all_genres)
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
multi_hot_labels = []
for genre_list in labels:
    y = np.zeros(n_classes, dtype=np.float32)
    for g in genre_list:
        y[genre_to_idx[g]] = 1.0
    multi_hot_labels.append(y)

# 33% для теста
from sklearn.model_selection import train_test_split
files = np.array(files)
multi_hot_labels = np.array(multi_hot_labels)
_, test_files, _, test_labels = train_test_split(files, multi_hot_labels, test_size=0.33, random_state=42)

# DataLoader
from Classes.CRNN_class import FMAMultiLabelDataset
BATCH_SIZE = 8
test_dataset = FMAMultiLabelDataset(test_files, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Загрузка модели
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(n_mels=128, n_classes=n_classes).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'crnn_fma_multiLabel.pth'), map_location=DEVICE))
model.eval()

# Тестирование
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb)
        preds_bin = (torch.sigmoid(preds) > 0.5).int().cpu().numpy()
        yb_bin = yb.int().cpu().numpy()
        all_preds.append(preds_bin)
        all_true.append(yb_bin)
all_preds = np.vstack(all_preds)
all_true = np.vstack(all_true)

print("F1 macro:", f1_score(all_true, all_preds, average='macro'))
print("F1 micro:", f1_score(all_true, all_preds, average='micro'))
print("Precision macro:", precision_score(all_true, all_preds, average='macro'))
print("Recall macro:", recall_score(all_true, all_preds, average='macro'))
print(classification_report(all_true, all_preds, target_names=all_genres))

# График F1-score, Precision, Recall по жанрам
f1s = f1_score(all_true, all_preds, average=None)
precisions = precision_score(all_true, all_preds, average=None)
recalls = recall_score(all_true, all_preds, average=None)

x = np.arange(len(all_genres))
plt.figure(figsize=(14, 6))
plt.bar(x - 0.2, f1s, width=0.2, label='F1-score')
plt.bar(x, precisions, width=0.2, label='Precision')
plt.bar(x + 0.2, recalls, width=0.2, label='Recall')
plt.xticks(x, all_genres, rotation=90)
plt.ylabel('Score')
plt.title('F1-score, Precision, Recall по жанрам')
plt.legend()
plt.tight_layout()
plt.show() 