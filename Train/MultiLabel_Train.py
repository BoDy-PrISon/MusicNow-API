import os
import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import Counter, defaultdict
import pickle
from Classes.CRNN_class import  CRNN ,FMAMultiLabelDataset

# Загрузка жанров
tracks = pd.read_csv('../fma_metadata/tracks.csv', header=[0, 1], index_col=0)
genres_df = pd.read_csv('../fma_metadata/genres.csv', index_col=0)
genre_id_to_name = dict(zip(genres_df.index, genres_df['title']))

# Для каждого трека получаем список жанров
files = []
labels = []
all_genres = set()
for idx, row in tracks.iterrows():
    tid = '{:06d}'.format(idx)
    path = os.path.join('E:/fma_large', tid[:3], tid + '.mp3')
    if not os.path.exists(path):
        continue
    genre_ids = row['track', 'genres']
    if pd.isna(genre_ids) or genre_ids == '[]':
        continue
    genre_ids = json.loads(genre_ids.replace("'", '"'))  # Преобразуем строку в список
    genre_names = [genre_id_to_name[g] for g in genre_ids if g in genre_id_to_name]
    if not genre_names:
        continue
    files.append(path)
    labels.append(genre_names)
    all_genres.update(genre_names)

# Список всех жанров
all_genres = sorted(list(all_genres))
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
n_classes = len(all_genres)

# === ДОБАВИТЬ СОЗДАНИЕ LabelEncoder ===
le = LabelEncoder()
le.fit(all_genres)
# === КОНЕЦ ДОБАВЛЕНИЯ ===

# Multi-hot encoding
multi_hot_labels = []
for genre_list in labels:
    y = np.zeros(n_classes, dtype=np.float32)
    for g in genre_list:
        y[genre_to_idx[g]] = 1.0
    multi_hot_labels.append(y)

# ОГРАНИЧЕНИЕ РАЗМЕРА ДАТАСЕТА
# N = 2000  # или любое другое число для быстрой проверки
# if len(files) > N:
    #dxs = random.sample(range(len(files)), N)
    #files = [files[i] for i in idxs]
    #labels = [labels[i] for i in idxs]

# Подсчёт количества треков для каждого жанра
label_counts = defaultdict(int)
for genre_list in labels:
    for g in genre_list:
        label_counts[g] += 1

# Фильтрация: оставляем только треки, где хотя бы один жанр встречается >=2 раз
files_filtered = []
labels_filtered = []
multi_hot_labels_filtered = []
for f, l, mh in zip(files, labels, multi_hot_labels):
    if any(label_counts[g] >= 2 for g in l):
        files_filtered.append(f)
        labels_filtered.append(l)
        multi_hot_labels_filtered.append(mh)
files = files_filtered
labels = labels_filtered
multi_hot_labels = multi_hot_labels_filtered

#Train/test split
files = np.array(files)
multi_hot_labels = np.array(multi_hot_labels)
train_files, test_files, train_labels, test_labels = train_test_split(
    files, multi_hot_labels, test_size=0.2, random_state=42)

#DataLoader
BATCH_SIZE = 8
train_dataset = FMAMultiLabelDataset(train_files, train_labels)
test_dataset = FMAMultiLabelDataset(test_files, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#Обучение и тестирование
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Используется устройство:", DEVICE)
if DEVICE.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))
model = CRNN(n_mels=128, n_classes=n_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
EPOCHS = 20
train_acc_list = []
test_acc_list = []

def multilabel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).astype(int)
    return (y_true == y_pred).mean()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds_bin = (torch.sigmoid(preds) > 0.5).int().cpu().numpy()
        yb_bin = yb.int().cpu().numpy()
        correct += (preds_bin == yb_bin).mean() * xb.size(0)
        total += xb.size(0)
    acc = correct / total
    train_acc_list.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/total:.4f} | Acc: {acc:.4f}")

    # Тест на каждой эпохе
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            preds_bin = (torch.sigmoid(preds) > 0.5).int().cpu().numpy()
            yb_bin = yb.int().cpu().numpy()
            correct += (preds_bin == yb_bin).mean() * xb.size(0)
            total += xb.size(0)
    test_acc = correct / total
    test_acc_list.append(test_acc)
    print(f"Test accuracy: {test_acc:.4f}")

# Сохранение модели и энкодера
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, 'Models')
os.makedirs(models_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(models_dir, 'crnn_fma_multiLabel.pth'))
with open(os.path.join(models_dir, 'genre_label_encoder_multiLabel.pkl'), 'wb') as f:
    pickle.dump(le, f)
print('Модель и энкодер жанров сохранены.')

# Если сохраняется список всех жанров:
with open(os.path.join(models_dir, 'all_genres_multiLabel.pkl'), 'wb') as f:
    pickle.dump(all_genres, f)
print('Список всех жанров сохранён в Models/all_genres_multiLabel.pkl')

# 10. Графики
# (a) Матрица ошибок
all_preds = []
all_true = []
model.eval()
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="Metrics"):
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

# (b) График точности по эпохам
plt.figure()
plt.plot(train_acc_list, label='Train Acc')
plt.plot(test_acc_list, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy per Epoch')
plt.show()

with open('../Models/all_genres.pkl', 'rb') as f:
    all_genres = pickle.load(f)

