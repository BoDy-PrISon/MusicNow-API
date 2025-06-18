import os
from collections import Counter
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
from Classes.CRNN_class import CRNN
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import hashlib


def create_cache_filename(file_path, n_mels=128, duration=30, sr=22050):
    """Создает уникальное имя для кэш файла на основе параметров"""
    # Создаем хэш от пути файла и параметров
    params_str = f"{file_path}_{n_mels}_{duration}_{sr}"
    hash_obj = hashlib.md5(params_str.encode())
    return hash_obj.hexdigest() + ".npy"


def preprocess_fma_to_cache(files, labels, cache_dir='../preprocessed_cache', n_mels=128, duration=30, sr=22050):
    """Предварительная обработка FMA треков в mel-спектрограммы"""
    
    print("=" * 70)
    print("    PREPROCESSING FMA - СОЗДАНИЕ MEL-КЭША")
    print("=" * 70)
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Проверяем сколько уже есть в кэше
    existing_files = set(os.listdir(cache_dir))
    cached_count = len([f for f in existing_files if f.endswith('.npy')])
    
    print(f"[INFO] Всего файлов для обработки: {len(files)}")
    print(f"[INFO] Уже в кэше: {cached_count}")
    print(f"[INFO] Параметры: n_mels={n_mels}, duration={duration}s, sr={sr}")
    
    # Создаем маппинг файл -> кэш
    cache_mapping = {}
    files_to_process = []
    
    for i, (file_path, label) in enumerate(zip(files, labels)):
        cache_filename = create_cache_filename(file_path, n_mels, duration, sr)
        cache_path = os.path.join(cache_dir, cache_filename)
        
        cache_mapping[file_path] = {
            'cache_path': cache_path,
            'label': label,
            'cache_filename': cache_filename
        }
        
        if not os.path.exists(cache_path):
            files_to_process.append((file_path, cache_path, label))
    
    if files_to_process:
        print(f"[PROCESSING] Нужно обработать: {len(files_to_process)} файлов")
        
        # Обработка с прогресс-баром
        failed_files = []
        samples = sr * duration
        
        for file_path, cache_path, label in tqdm(files_to_process, desc="Создание mel-кэша"):
            try:
                # Загружаем аудио
                y, _ = librosa.load(file_path, sr=sr, duration=duration)
                
                # Padding/truncating
                if len(y) < samples:
                    y = np.pad(y, (0, samples - len(y)))
                else:
                    y = y[:samples]
                
                # Mel-спектрограмма
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Нормализация
                mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
                
                # Сохраняем
                np.save(cache_path, mel_db.astype(np.float32))
                
            except Exception as e:
                print(f"[ERROR] Ошибка обработки {file_path}: {e}")
                failed_files.append(file_path)
                continue
        
        if failed_files:
            print(f"[WARNING] Не удалось обработать {len(failed_files)} файлов")
        
        print(f"[SUCCESS] Preprocessing завершен!")
    else:
        print("[INFO] Все файлы уже в кэше!")
    
    # Сохраняем маппинг
    mapping_path = os.path.join(cache_dir, 'file_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(cache_mapping, f)
    
    # Статистика кэша
    cache_size_mb = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                       for f in os.listdir(cache_dir) if f.endswith('.npy')) / (1024**2)
    
    print(f"[CACHE STATS]")
    print(f"  • Размер кэша: {cache_size_mb:.1f} MB")
    print(f"  • Файлов в кэше: {len([f for f in os.listdir(cache_dir) if f.endswith('.npy')])}")
    print(f"  • Экономия места: ~{100 - (cache_size_mb/1024):.0f}%")
    
    return cache_mapping


# ВЫНЕСЕН НА УРОВЕНЬ МОДУЛЯ для Windows multiprocessing
class FMAMultiClassDatasetCached(Dataset):
    def __init__(self, files, labels, cache_mapping):
        self.files = files
        self.labels = labels
        self.cache_mapping = cache_mapping
        self.target_frames = 1293  # Фиксированный размер

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Загружаем из кэша вместо декодирования MP3!
            cache_path = self.cache_mapping[file_path]['cache_path']
            mel_db = np.load(cache_path)
            
            # ФИКСИМ РАЗМЕР - это важно!
            mel_db = self._fix_mel_size(mel_db)
            
            mel_db = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
            return mel_db, label
            
        except Exception as e:
            print(f'[ERROR] Ошибка при загрузке кэша {cache_path}: {e}')
            # Fallback к прямой загрузке
            return self._load_direct(file_path, label)
    
    def _fix_mel_size(self, mel_db):
        """Фиксирует размер mel-спектрограммы до target_frames"""
        current_frames = mel_db.shape[1]
        
        if current_frames < self.target_frames:
            # Padding справа
            pad_width = self.target_frames - current_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_db.min())
        elif current_frames > self.target_frames:
            # Обрезаем
            mel_db = mel_db[:, :self.target_frames]
        
        return mel_db
    
    def _load_direct(self, file_path, label):
        """Fallback метод прямой загрузки"""
        try:
            y, sr = librosa.load(file_path, sr=22050, duration=30)
            
            # ФИКСИРОВАННАЯ длительность - это ключ!
            samples = 22050 * 30  # Ровно 30 секунд
            if len(y) < samples:
                y = np.pad(y, (0, samples - len(y)))
            else:
                y = y[:samples]
            
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            
            # Фиксим размер и здесь тоже
            mel_db = self._fix_mel_size(mel_db)
            
            return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0), label
        except Exception as e:
            print(f'[ERROR] Полная ошибка загрузки {file_path}: {e}')
            # Возвращаем тензор фиксированного размера
            return torch.zeros(1, 128, self.target_frames), label


def analyze_fma_dataset(files, labels, title_suffix=""):
    """Анализ и визуализация датасета FMA"""
    
    print("=" * 70)
    print(f"   АНАЛИЗ ДАТАСЕТА FMA - КЛАССИФИКАЦИЯ ЖАНРОВ {title_suffix}  ")
    print("=" * 70)
    
    genre_counts = Counter(labels)
    
    print(f"Общее количество треков: {len(files)}")
    print(f"Количество жанров: {len(genre_counts)}")
    
    # Создание графиков
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Столбчатая диаграмма жанров
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    
    axes[0,0].bar(genres, counts, color='lightcoral', alpha=0.7)
    axes[0,0].set_title(f'Распределение жанров в FMA {title_suffix}')
    axes[0,0].set_xlabel('Жанры')
    axes[0,0].set_ylabel('Количество треков')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Круговая диаграмма
    axes[0,1].pie(counts, labels=genres, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title(f'Процентное распределение жанров {title_suffix}')
    
    # 3. Горизонтальная диаграмма (отсортированная)
    sorted_items = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_genres = [item[0] for item in sorted_items]
    sorted_counts = [item[1] for item in sorted_items]
    
    axes[0,2].barh(sorted_genres, sorted_counts, color='lightblue', alpha=0.7)
    axes[0,2].set_title('Жанры (отсортировано)')
    axes[0,2].set_xlabel('Количество треков')
    
    # 4. Пример длительностей (симуляция)
    durations = np.random.normal(180, 60, len(files))  # Примерные длительности
    axes[1,0].hist(durations, bins=30, alpha=0.7, color='orange')
    axes[1,0].set_title('Распределение длительности треков (примерное)')
    axes[1,0].set_xlabel('Длительность (сек)')
    axes[1,0].set_ylabel('Количество треков')
    
    # 5. Временной тренд (симуляция)
    years = np.random.choice(range(1950, 2020), len(files))
    year_counts = Counter(years)
    sorted_years = sorted(year_counts.keys())
    year_values = [year_counts[year] for year in sorted_years]
    
    axes[1,1].plot(sorted_years, year_values, marker='o', alpha=0.7)
    axes[1,1].set_title('Распределение по годам (примерное)')
    axes[1,1].set_xlabel('Год')
    axes[1,1].set_ylabel('Количество треков')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Статистика дисбаланса
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    avg_duration = np.mean(durations)
    
    axes[1,2].text(0.1, 0.9, f'Статистика FMA {title_suffix}:', fontsize=14, fontweight='bold', transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.8, f'Всего треков: {len(files)}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.7, f'Жанров: {len(genre_counts)}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.6, f'Макс. треков: {max_count}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.5, f'Мин. треков: {min_count}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.4, f'Коэф. дисбаланса: {imbalance_ratio:.2f}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.3, f'Среднее: {np.mean(counts):.1f}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.2, f'Медиана: {np.median(counts):.1f}', fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.1, f'Средняя длительность: {avg_duration:.0f}с', fontsize=12, transform=axes[1,2].transAxes)
    
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    # Сохранение с разными именами
    if title_suffix == "(ПОЛНЫЙ)":
        plt.savefig('../Models/fma_dataset_analysis_full.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('../Models/fma_dataset_analysis_limited.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Детальная статистика
    print("\nДетальное распределение жанров:")
    for genre, count in sorted_items:
        percentage = (count / len(files)) * 100
        print(f"  {genre}: {count} треков ({percentage:.1f}%)")
    
    # Рекомендации
    print(f"\nРекомендации по балансировке:")
    if imbalance_ratio > 5:
        print("  [WARNING] Очень высокий дисбаланс!")
        print("     - Обязательно используйте class_weight='balanced'")
        print("     - Рассмотрите oversampling/undersampling")
    elif imbalance_ratio > 2:
        print("  [WARNING] Умеренный дисбаланс")
        print("     - Используйте взвешенную функцию потерь")
    else:
        print("  [OK] Датасет хорошо сбалансирован")
    
    return genre_counts


def train_fma_model():
    print("=" * 70)
    print("           FMA TRAINING - КЛАССИФИКАЦИЯ ЖАНРОВ")
    print("=" * 70)

    # STEP 1: Загрузка ВСЕХ данных для анализа
    print("\n[STEP 1] Загрузка метаданных FMA...")
    tracks = pd.read_csv('../fma_metadata/tracks.csv', header=[0, 1], index_col=0)
    all_files = []
    all_labels = []
    
    for idx, row in tracks.iterrows():
        tid = '{:06d}'.format(idx)
        path = os.path.join('E:/fma_large', tid[:3], tid + '.mp3')
        genre = row['track', 'genre_top']
        if not os.path.exists(path):
            continue
        if pd.isna(genre):
            continue
        all_files.append(path)
        all_labels.append(genre)

    print(f"[INFO] Найдено треков в полном датасете: {len(all_files)}")

    # Базовая фильтрация (убираем жанры с <2 треками)
    print("\n[STEP 2] Базовая фильтрация полного датасета...")
    label_counts = Counter(all_labels)
    files_filtered = []
    labels_filtered = []
    for f, l in zip(all_files, all_labels):
        if label_counts[l] >= 2:
            files_filtered.append(f)
            labels_filtered.append(l)
    
    full_files = files_filtered
    full_labels = labels_filtered
    print(f"[INFO] После фильтрации: {len(full_files)} треков")

    # STEP 2.5: PREPROCESSING - создание mel-кэша
    print("\n[STEP 2.5] Создание/проверка mel-кэша...")
    cache_mapping = preprocess_fma_to_cache(full_files, full_labels)
    
    # Пауза после preprocessing
    input("\n[PAUSE] Preprocessing завершен. Нажмите Enter для анализа датасета...")

    # АНАЛИЗ ПОЛНОГО ДАТАСЕТА
    print("\n[ANALYSIS] Анализ ПОЛНОГО датасета:")
    full_stats = analyze_fma_dataset(full_files, full_labels, "(ПОЛНЫЙ)")
    
    # Пауза для изучения полного анализа
    input("\n[PAUSE] Изучите анализ ПОЛНОГО датасета. Нажмите Enter для начала обучения...")

    # Используем ПОЛНЫЙ датасет для обучения
    files_final = full_files
    labels_final = full_labels
    print(f"[INFO] Используем ПОЛНЫЙ датасет для обучения: {len(files_final)} треков")

    # Кодирование жанров
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_final)
    num_classes = len(le.classes_)
    print(f"[INFO] Количество жанров для обучения: {num_classes}")

    print("\n" + "=" * 70)
    print("                  НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)

    # Логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'../logs/fma_train_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Train/test split
    print("\n[STEP 4] Разделение на train/test...")
    train_files, test_files, train_labels, test_labels = train_test_split(
        files_final, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)
    
    print(f"  • Train: {len(train_files)} треков")
    print(f"  • Test: {len(test_files)} треков")

    # ОПТИМИЗИРОВАННЫЕ DataLoader'ы БЕЗ multiprocessing (проще и стабильнее)
    BATCH_SIZE = 24  # Увеличен для экономии VRAM
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n[STEP 5] Создание ОПТИМИЗИРОВАННОЙ модели...")
    print(f"[INFO] Используется устройство: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OPTIMIZATION] Mixed precision: Включено")
        print(f"[OPTIMIZATION] Batch size: {BATCH_SIZE}")

    train_dataset = FMAMultiClassDatasetCached(train_files, train_labels, cache_mapping)
    test_dataset = FMAMultiClassDatasetCached(test_files, test_labels, cache_mapping)
    
    # ОПТИМИЗИРОВАННЫЕ DataLoader'ы БЕЗ multiprocessing (проще и стабильнее)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Убираем multiprocessing
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,  # Убираем multiprocessing
        pin_memory=True
    )

    model = CRNN(n_mels=128, n_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  • Параметров модели: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Без mixed precision - просто и стабильно
    # scaler = GradScaler()  # Закомментировать
    
    # Сохранение конфигурации
    config = {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': 1e-3,
        'model': 'CRNN',
        'dataset': 'FMA_FULL',  # Указываем что используем полный датасет
        'num_classes': num_classes,
        'optimizations': ['cached_preprocessing', 'mixed_precision', 'optimized_dataloader'],
        'full_dataset_stats': dict(full_stats),
        # 'limited_dataset_stats': dict(limited_stats)  # Закомментировано
    }
    
    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ОБУЧЕНИЕ С ОПТИМИЗАЦИЯМИ
    print("\n[TRAINING] Обучение...")
    print("=" * 50)
    train_acc_list = []
    test_acc_list = []

    for epoch in range(EPOCHS):
        # Training phase - добавляем инициализацию переменных
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{EPOCHS} [Train]"):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Обычный forward pass
            preds = model(xb)
            loss = criterion(preds, yb)
            
            # Обычный backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)
        
        train_acc = correct / total
        train_loss = total_loss / total
        train_acc_list.append(train_acc)

        # Testing phase
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_true = []
        test_loss = 0
        
        with torch.no_grad():
            for xb, yb in tqdm(test_loader, desc=f"Epoch {epoch+1:2d}/{EPOCHS} [Test]"):
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                
                # Убираем autocast и из testing
                preds = model(xb)
                loss = criterion(preds, yb)
                
                test_loss += loss.item() * xb.size(0)
                all_preds.extend(preds.argmax(1).cpu().numpy())
                all_true.extend(yb.cpu().numpy())
                correct += (preds.argmax(1) == yb).sum().item()
                total += xb.size(0)
        
        test_acc = correct / total
        test_loss = test_loss / total
        test_acc_list.append(test_acc)

        print(f'\n[EPOCH {epoch + 1:2d}/{EPOCHS}] RESULTS:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Test Loss : {test_loss:.4f} | Test Acc : {test_acc:.4f}')

        # Логирование
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)

        print('-' * 50)

    writer.close()

    # Сохранение модели и энкодера
    print(f"\n[SAVING] Сохранение модели...")
    os.makedirs('../Models', exist_ok=True)
    torch.save(model.state_dict(), '../Models/crnn_fma_multiclass.pth')
    with open('../Models/genre_label_encoder_multiclass.pkl', 'wb') as f:
        pickle.dump(le, f)
    print('[SUCCESS] Модель и энкодер жанров сохранены.')

    # ФИНАЛЬНЫЕ ГРАФИКИ И МЕТРИКИ
    print(f"\n[RESULTS] Построение финальных графиков...")
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_true, all_preds, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation='vertical')
    plt.title('Confusion Matrix - FMA Genre Classification')
    plt.tight_layout()
    plt.savefig('../Models/fma_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Classification Report
    print(f"\n[CLASSIFICATION REPORT]")
    print("=" * 50)
    print(classification_report(all_true, all_preds, target_names=le.classes_))

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(range(1, EPOCHS+1), test_acc_list, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Progress - FMA Genre Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../Models/fma_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Финальные результаты
    final_train_acc = train_acc_list[-1]
    final_test_acc = test_acc_list[-1]
    
    print("\n" + "=" * 70)
    print("                    ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"[DATASET] ПОЛНЫЙ датасет использован: {len(files_final)} треков")
    print(f"[RESULT] Финальная Train Accuracy: {final_train_acc:.4f}")
    print(f"[RESULT] Финальная Test Accuracy : {final_test_acc:.4f}")
    print(f"[OUTPUT] Модель сохранена: ../Models/crnn_fma_multiclass.pth")
    print(f"[OUTPUT] Энкодер сохранен: ../Models/genre_label_encoder_multiclass.pkl")
    print(f"[LOGS]   TensorBoard логи: {log_dir}")
    print("=" * 70)


if __name__ == "__main__":
    train_fma_model() 