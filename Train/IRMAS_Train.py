import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, hamming_loss, jaccard_score
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
from collections import Counter

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classes.IRMAS_dataset import IRMASDataset, IRMAS_INSTRUMENTS
from Classes.CRNN_class import CRNN_IRMAS


def train_irmas_model():
    # Параметры
    DATA_DIR = "D:/IRMAS-TrainingData"
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("          IRMAS TRAINING - РАСПОЗНАВАНИЕ ИНСТРУМЕНТОВ")
    print("=" * 70)
    print(f"[INFO] Используем устройство: {DEVICE}")

    # Загружаем датасет
    print("\n[STEP 1] Загружаем датасет...")
    dataset = IRMASDataset(DATA_DIR)
    print(f"[SUCCESS] Датасет загружен: {len(dataset)} сэмплов")

    print("\n" + "=" * 70)
    print("                    НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)

    # Логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'../logs/irmas_train_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Сохранение конфигурации
    config = {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'model': 'CRNN_IRMAS',
        'dataset': 'IRMAS',
    }
    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Разделяем на train/validation (80/20)
    print("\n[STEP 2] Разделение на train/validation...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"  • Train: {train_size} сэмплов")
    print(f"  • Validation: {val_size} сэмплов")

    # Создаём DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Создаём модель
    print("\n[STEP 3] Создание модели CRNN...")
    model = CRNN_IRMAS(n_mels=128, n_classes=11).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  • Параметров модели: {total_params:,}")

    # Loss function для multi-label (BCELoss, так как уже есть sigmoid)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("\n[TRAINING] Начинается обучение...")
    print("=" * 50)
    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1:2d}/{EPOCHS}, Batch {batch_idx:3d}/{len(train_loader)}, Loss: {loss.item():.4f}')

        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
                predictions = (output > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'\n[EPOCH {epoch + 1:2d}/{EPOCHS}] RESULTS:')
        print(f'  Train Loss     : {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        print(f'  F1 Macro       : {f1_macro:.4f}')
        print(f'  Accuracy       : {accuracy:.4f}')

        if f1_macro > best_f1:
            best_f1 = f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), '../Models/crnn_irmas_instruments.pth')
            print(f'  [SAVED] Новая лучшая модель (F1: {f1_macro:.4f})')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"\n[EARLY STOP] Остановка на эпохе {epoch + 1}")
            break
        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('F1/Validation', f1_macro, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        train_acc_list.append(accuracy)
        val_acc_list.append(accuracy)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        print('-' * 50)

    writer.close()
    print("\n" + "=" * 70)
    print("                      ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"[RESULT] Лучший F1 Score: {best_f1:.4f}")
    print(f"[OUTPUT] Модель сохранена: ../Models/crnn_irmas_instruments.pth")
    print(f"[LOGS]   TensorBoard логи: {log_dir}")
    print("=" * 70)

    # --- ОСТАВЛЯЕМ ТОЛЬКО ДВА ГРАФИКА ---
    # 1. Комбинированный график истории обучения (accuracy и loss)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(range(1, len(train_acc_list)+1), train_acc_list, label='Train Accuracy', color='blue', marker='o')
    ax1.plot(range(1, len(val_acc_list)+1), val_acc_list, label='Validation Accuracy', color='red', marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Train Loss', color='orange', marker='^', linestyle='--')
    ax2.plot(range(1, len(val_loss_list)+1), val_loss_list, label='Validation Loss', color='purple', marker='v', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    plt.title('IRMAS - Training Progress (Accuracy & Loss)')
    plt.tight_layout()
    plt.show()

    # 2. График метрик по инструментам
    instrument_metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    for i, instrument in enumerate(IRMAS_INSTRUMENTS):
        y_true_binary = all_targets[:, i]
        y_pred_binary = (all_predictions[:, i] > 0.5).astype(int)
        report = classification_report(y_true_binary, y_pred_binary, 
                                     target_names=['Absent', 'Present'], 
                                     output_dict=True, zero_division=0)
        instrument_metrics['precision'].append(report['Present']['precision'])
        instrument_metrics['recall'].append(report['Present']['recall'])
        instrument_metrics['f1'].append(report['Present']['f1-score'])
    x = np.arange(len(IRMAS_INSTRUMENTS))
    width = 0.25
    plt.figure(figsize=(15, 8))
    plt.bar(x - width, instrument_metrics['precision'], width, label='Precision', alpha=0.8)
    plt.bar(x, instrument_metrics['recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, instrument_metrics['f1'], width, label='F1-Score', alpha=0.8)
    plt.xlabel('Instruments')
    plt.ylabel('Score')
    plt.title('IRMAS - Performance Metrics by Instrument')
    plt.xticks(x, [inst.replace('_', ' ').title() for inst in IRMAS_INSTRUMENTS], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_irmas_model()