import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Classes.CRNN_class import CRNN
import librosa
from tqdm import tqdm
import librosa.effects as effects
import torch.nn.functional as F
import time

# Import for classification and metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, balanced_accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


AUDIO_DIR = 'D:/DEAM/deam-mediaeval-dataset-emotional-analysis-in-music/versions/1/DEAM_audio/MEMD_audio'
ANNOTATIONS_CSV = 'D:/DEAM/deam-mediaeval-dataset-emotional-analysis-in-music/versions/1/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv'

SEGMENT_SECONDS = 20
STEP_SECONDS = 10  # 50% overlap
SR = 22050
N_MELS = 128

# 1. Load annotations and convert to categories
df = pd.read_csv(ANNOTATIONS_CSV)
df.columns = df.columns.str.strip()

# После загрузки аннотаций добавьте:
print(f"Всего песен в аннотациях: {len(df)}")
print(f"Диапазон arousal: {df['arousal_mean'].min():.2f} - {df['arousal_mean'].max():.2f}")
print(f"Диапазон valence: {df['valence_mean'].min():.2f} - {df['valence_mean'].max():.2f}")
print(f"Средние значения: arousal={df['arousal_mean'].mean():.2f}, valence={df['valence_mean'].mean():.2f}")

# Добавьте этот блок временно для анализа распределения
print("\nАнализ распределения Arousal и Valence:")
print(df[['arousal_mean', 'valence_mean']].describe())

# ТЕПЕРЬ вычисляем медианы (после загрузки данных!)
arousal_median = df['arousal_mean'].median()
valence_median = df['valence_mean'].median()
print(f"\nМедианы: arousal={arousal_median:.2f}, valence={valence_median:.2f}")

# Обновленная функция с правильными значениями по умолчанию
def get_mood_category(arousal, valence, arousal_thresh=None, valence_thresh=None):
    if arousal_thresh is None:
        arousal_thresh = arousal_median
    if valence_thresh is None:
        valence_thresh = valence_median
        
    if arousal >= arousal_thresh and valence >= valence_thresh:
        return 'Joy_Excitement'      # Высокое A, Высокое V
    elif arousal >= arousal_thresh and valence < valence_thresh:
        return 'Anger_Fear'          # Высокое A, Низкое V
    elif arousal < arousal_thresh and valence >= valence_thresh:
        return 'Calm_Peaceful'       # Низкое A, Высокое V
    elif arousal < arousal_thresh and valence < valence_thresh:
        return 'Sadness_Melancholy'  # Низкое A, Низкое V
    else:
        return 'Unknown'

def recommend_threshold(df):
    """Рекомендация оптимальных порогов"""
    
    # Цель: сбалансированное распределение категорий
    mean_arousal = df['arousal_mean'].mean()
    mean_valence = df['valence_mean'].mean()
    median_arousal = df['arousal_mean'].median()
    median_valence = df['valence_mean'].median()
    
    print("\n=== РЕКОМЕНДАЦИИ ===")
    print(f"Для сбалансированного распределения рекомендуется:")
    print(f"1. Медиана: Arousal={median_arousal:.2f}, Valence={median_valence:.2f}")
    print(f"2. Среднее: Arousal={mean_arousal:.2f}, Valence={mean_valence:.2f}")
    
    # Проверка стандартного отклонения
    std_arousal = df['arousal_mean'].std()
    std_valence = df['valence_mean'].std()
    
    print(f"\nСтандартные отклонения:")
    print(f"Arousal: {std_arousal:.2f}")
    print(f"Valence: {std_valence:.2f}")
    
    # Если данные нормально распределены, используйте среднее
    # Если есть выбросы, используйте медиану
    if std_arousal < 1.5 and std_valence < 1.5:
        print("\nРЕКОМЕНДАЦИЯ: Используйте СРЕДНЕЕ (данные хорошо распределены)")
        return mean_arousal, mean_valence
    else:
        print("\nРЕКОМЕНДАЦИЯ: Используйте МЕДИАНУ (есть выбросы)")
        return median_arousal, median_valence

# ВЫЗВАТЬ ФУНКЦИЮ ЗДЕСЬ (перед использованием переменных)
recommended_arousal, recommended_valence = 5.0, 5.0  # Фиксированные пороги
print(f"\nИТОГОВЫЕ ПОРОГИ: Arousal={recommended_arousal:.2f}, Valence={recommended_valence:.2f}")

# Добавьте после загрузки данных для анализа оптимальных порогов
def analyze_thresholds(df):
    """Анализ распределения и подбор оптимальных порогов"""
    
    # 1. Базовая статистика
    print("=== АНАЛИЗ РАСПРЕДЕЛЕНИЯ ДАННЫХ ===")
    print(f"Количество треков: {len(df)}")
    print("\nСтатистика Arousal:")
    print(df['arousal_mean'].describe())
    print("\nСтатистика Valence:")
    print(df['valence_mean'].describe())
    
    # 2. Визуализация распределения
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Гистограммы
    axes[0,0].hist(df['arousal_mean'], bins=30, alpha=0.7, color='red')
    axes[0,0].axvline(df['arousal_mean'].mean(), color='blue', linestyle='--', label=f'Mean: {df["arousal_mean"].mean():.2f}')
    axes[0,0].axvline(df['arousal_mean'].median(), color='green', linestyle='--', label=f'Median: {df["arousal_mean"].median():.2f}')
    axes[0,0].set_title('Arousal Distribution')
    axes[0,0].legend()
    
    axes[0,1].hist(df['valence_mean'], bins=30, alpha=0.7, color='blue')
    axes[0,1].axvline(df['valence_mean'].mean(), color='red', linestyle='--', label=f'Mean: {df["valence_mean"].mean():.2f}')
    axes[0,1].axvline(df['valence_mean'].median(), color='green', linestyle='--', label=f'Median: {df["valence_mean"].median():.2f}')
    axes[0,1].set_title('Valence Distribution')
    axes[0,1].legend()
    
    # Scatter plot
    axes[1,0].scatter(df['valence_mean'], df['arousal_mean'], alpha=0.6)
    axes[1,0].axhline(df['arousal_mean'].median(), color='red', linestyle='--', label=f'Arousal median: {df["arousal_mean"].median():.2f}')
    axes[1,0].axvline(df['valence_mean'].median(), color='blue', linestyle='--', label=f'Valence median: {df["valence_mean"].median():.2f}')
    axes[1,0].set_xlabel('Valence')
    axes[1,0].set_ylabel('Arousal')
    axes[1,0].set_title('Arousal-Valence Space')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Тестирование разных порогов
    thresholds_to_test = [
        (df['arousal_mean'].mean(), df['valence_mean'].mean(), "Mean"),
        (df['arousal_mean'].median(), df['valence_mean'].median(), "Median"),
        (5.0, 5.0, "Fixed 5.0"),
        (df['arousal_mean'].quantile(0.6), df['valence_mean'].quantile(0.6), "60th percentile")
    ]
    
    print("\n=== ТЕСТИРОВАНИЕ ПОРОГОВ ===")
    results = []
    
    for arousal_thresh, valence_thresh, name in thresholds_to_test:
        categories = []
        for _, row in df.iterrows():
            category = get_mood_category(row['arousal_mean'], row['valence_mean'], 
                                       arousal_thresh, valence_thresh)
            categories.append(category)
        
        category_counts = Counter(categories)
        print(f"\n{name} (A={arousal_thresh:.2f}, V={valence_thresh:.2f}):")
        for cat, count in category_counts.items():
            percentage = (count / len(categories)) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")
        
        results.append((name, category_counts, arousal_thresh, valence_thresh))
    
    # Визуализация результатов
    categories = ['Joy_Excitement', 'Anger_Fear', 'Calm_Peaceful', 'Sadness_Melancholy']
    x = np.arange(len(categories))
    width = 0.2
    
    axes[1,1].clear()
    for i, (name, counts, _, _) in enumerate(results):
        values = [counts.get(cat, 0) for cat in categories]
        axes[1,1].bar(x + i*width, values, width, label=name)
    
    axes[1,1].set_xlabel('Mood Categories')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Category Distribution by Threshold')
    axes[1,1].set_xticks(x + width * 1.5)
    axes[1,1].set_xticklabels(categories, rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('../Models/threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# Запуск анализа
threshold_results = analyze_thresholds(df)

# Также добавьте в начало файла функцию для красивого вывода:

USE_AUGMENTATION = True  # Добавьте эту строку!

def print_progress_header():
    print("=" * 70)
    print(" МУЗЫКАЛЬНЫЙ АНАЛИЗАТОР НАСТРОЕНИЯ - ОБРАБОТКА ДАННЫХ ")
    print("=" * 70)
    print("  Этапы обработки:")
    print("  Загрузка аудиофайлов")
    print("  Создание мел-спектрограмм")
    print("  Аугментация для редких классов")
    print("  Кодирование меток")
    print("=" * 70)

# Вызовите перед основным циклом:
print_progress_header()

# Теперь основной цикл обработки с прогресс-барами БЕЗ СМАЙЛИКОВ
X_by_song = {}
y_by_song = {}
total_augmented_samples = 0

print(f"\nНачинаем обработку {len(df)} песен...")

start_time = time.time()

# Основной прогресс-бар для песен
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка песен", ncols=100):
    song_id = int(row['song_id'])
    audio_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")
    if not os.path.exists(audio_path):
        continue
    
    try:
        y, sr = librosa.load(audio_path, sr=SR)
    except Exception as e:
        tqdm.write(f"Error loading {audio_path}: {e}")
        continue
    
    song_segments = []
    song_labels_categories = []
    total_samples = len(y)
    segment_samples = SEGMENT_SECONDS * SR
    step_samples = STEP_SECONDS * SR
    
    # Определяем метку настроения
    arousal_mean = row['arousal_mean']
    valence_mean = row['valence_mean']
    mood_category = get_mood_category(arousal_mean, valence_mean, recommended_arousal, recommended_valence)
    
    # Обработка сегментов с мини прогресс-баром
    segment_range = range(0, total_samples - segment_samples + 1, step_samples)
    
    for start in tqdm(segment_range, 
                      desc=f"Сегменты ({mood_category[:4]})",
                      leave=False, ncols=80):
        y_seg = y[start:start+segment_samples]
        
        # Основной сегмент
        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        song_segments.append(mel_db)
        song_labels_categories.append(mood_category)
        
        # Аугментация для малых классов
        if USE_AUGMENTATION and mood_category in ['Anger_Fear', 'Calm_Peaceful']:
            augmentation_factor = 2
            
            # Прогресс-бар для аугментации
            for aug_step in tqdm(range(augmentation_factor), 
                               desc=f"Аугментация {mood_category[:4]}",
                               leave=False, ncols=70):
                try:
                    # БЫСТРАЯ аугментация
                    y_aug = y_seg.copy()
                    
                    # 1. Добавление шума (очень быстро)
                    noise = np.random.normal(0, 0.005, len(y_aug))
                    y_aug = y_aug + noise
                    
                    # 2. Изменение громкости (очень быстро)
                    volume_factor = np.random.uniform(0.8, 1.2)
                    y_aug = y_aug * volume_factor
                    
                    # 3. Простой сдвиг во времени (очень быстро)
                    shift_samples = np.random.randint(-SR//10, SR//10)  # ±0.1 сек
                    y_aug = np.roll(y_aug, shift_samples)
                    
                    # Создаем мел-спектрограмму
                    mel_aug = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_mels=N_MELS)
                    mel_aug_db = librosa.power_to_db(mel_aug, ref=np.max)
                    mel_aug_db = (mel_aug_db - mel_aug_db.mean()) / (mel_aug_db.std() + 1e-6)
                    
                    song_segments.append(mel_aug_db)
                    song_labels_categories.append(mood_category)
                    total_augmented_samples += 1
                    
                except Exception as e:
                    tqdm.write(f"Augmentation error: {e}")

    # Filter out songs with no segments or unknown categories
    if song_segments and 'Unknown' not in song_labels_categories:
        X_by_song[song_id] = song_segments
        y_by_song[song_id] = song_labels_categories[0]
    
    # Периодическое обновление статистики
    if (idx + 1) % 100 == 0:
        current_counts = Counter(y_by_song.values())
        tqdm.write(f"Обработано {len(X_by_song)} песен. Распределение: {dict(current_counts)}")
        tqdm.write(f"Создано аугментированных сэмплов: {total_augmented_samples}")

print(f"\nОбработка завершена!")
print(f"Обработано песен: {len(X_by_song)}")
print(f"Создано аугментированных сэмплов: {total_augmented_samples}")

# Добавьте эти строки для финальной статистики:
category_counts_by_song = Counter(y_by_song.values())
print("\nРаспределение категорий настроения по песням:")
for category, count in category_counts_by_song.items():
    percentage = (count / len(y_by_song)) * 100
    print(f"  {category}: {count} ({percentage:.1f}%)")

print(f"Найдено аудиофайлов: {len(X_by_song)}")

# Encode mood categories
all_mood_categories = list(y_by_song.values())
le = LabelEncoder()
labels_encoded = le.fit_transform(all_mood_categories)
num_classes = len(le.classes_)
print(f"Found {num_classes} mood categories: {le.classes_}")

song_ids_filtered = list(y_by_song.keys())
labels_encoded_dict = dict(zip(song_ids_filtered, labels_encoded))


# 2. Split songs into train/val/test (using encoded labels for stratification)
song_ids = list(X_by_song.keys())
song_labels_for_split = [labels_encoded_dict[sid] for sid in song_ids]
train_ids, temp_ids, y_train_split, y_temp_split = train_test_split(song_ids, song_labels_for_split, test_size=0.3, random_state=42, stratify=song_labels_for_split)
val_ids, test_ids, y_val_split, y_test_split = train_test_split(temp_ids, y_temp_split, test_size=0.5, random_state=42, stratify=y_temp_split)

# 3. Form segment datasets with ONE label per song
X_train, y_train = [], []
for sid in train_ids:
    X_train.extend(X_by_song[sid])
    # Assign the same encoded mood label of the song to each segment
    song_encoded_label = labels_encoded_dict[sid]
    y_train.extend([song_encoded_label] * len(X_by_song[sid]))

X_val, y_val = [], []
for sid in val_ids:
    X_val.extend(X_by_song[sid])
    song_encoded_label = labels_encoded_dict[sid]
    y_val.extend([song_encoded_label] * len(X_by_song[sid]))

X_test, y_test = [], []
for sid in test_ids:
    X_test.extend(X_by_song[sid])
    song_encoded_label = labels_encoded_dict[sid]
    y_test.extend([song_encoded_label] * len(X_by_song[sid]))

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print(f"Size of training segment sample: {len(X_train)}")
print(f"Size of validation segment sample: {len(X_val)}")
print(f"Size of test segment sample: {len(X_test)}")

# 4. PyTorch Dataset for segments (now with a single numerical label)
class SegmentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
        y = torch.tensor(self.y[idx], dtype=torch.long)  # CHANGED: dtype to long for CrossEntropyLoss
        return x, y

BATCH_SIZE = 8
train_dataset = SegmentDataset(X_train, y_train)
val_dataset = SegmentDataset(X_val, y_val)
test_dataset = SegmentDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. CRNN Training (classification)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(n_mels=N_MELS, n_classes=num_classes).to(DEVICE) # CHANGED: n_classes = number of categories
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Вычисляем веса классов
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {class_weights}")

# Взвешенная функция потерь
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=1, gamma=2)

EPOCHS = 20

train_acc_list = []
val_acc_list = []

scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
best_val_acc = 0
patience_counter = 0
early_stopping_patience = 7

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += xb.size(0)
    acc = correct / total
    train_acc_list.append(acc)
    avg_loss = total_loss / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="Validation"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)
            # Collection for validation (optional, more important for test)
            # all_preds.extend(preds.argmax(1).cpu().numpy())
            # all_true.extend(yb.cpu().numpy())

    val_acc = correct / total
    val_acc_list.append(val_acc)
    print(f"  Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../Models/best_mood_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered!")
        break
    
    scheduler.step()

# 6. Evaluate on test set
model.eval()
correct = 0
total = 0
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="Testing"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        all_preds.extend(preds.argmax(1).cpu().numpy())
        all_true.extend(yb.cpu().numpy())
        correct += (preds.argmax(1) == yb).sum().item()
        total += xb.size(0)

test_acc = correct / total
print(f"\nTest accuracy: {test_acc:.4f}")

# После тестирования добавьте:
balanced_acc = balanced_accuracy_score(all_true, all_preds)
kappa = cohen_kappa_score(all_true, all_preds)

print(f"Balanced accuracy: {balanced_acc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")

# 7. Save model and encoder
os.makedirs('../Models', exist_ok=True)
torch.save(model.state_dict(), '../Models/crnn_deam_mood_classification.pth')
with open('../Models/mood_label_encoder_classification.pkl', 'wb') as f:
    pickle.dump(le, f)
print('Mood classification model and encoder saved.')

# 8. Metrics for classification
print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=le.classes_))

cm = confusion_matrix(all_true, all_preds, labels=range(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation='vertical')
plt.title('Confusion Matrix (Mood Classification)')
plt.tight_layout()
plt.savefig('../Models/mood_confusion_matrix.png') # Save confusion matrix to file
plt.show()

# 9. Accuracy plots
plt.figure()
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy per Epoch (Mood Classification)')
plt.savefig('../Models/mood_accuracy_plot.png') # Save accuracy plot to file
plt.show()

end_time = time.time()
processing_time = end_time - start_time

print(f"\nВремя обработки: {processing_time/60:.1f} минут")
print(f"Скорость: {len(X_by_song)/processing_time:.1f} песен/сек")
if total_augmented_samples > 0:
    print(f"Скорость аугментации: {total_augmented_samples/processing_time:.1f} сэмплов/сек")
