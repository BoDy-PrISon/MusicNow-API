import os
import pickle
import torch
import librosa
import sys

# Добавляем корневую директорию проекта в sys.path, чтобы импортировать модули recognizers и Classes
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from recognizers.deam_infer import infer_mood_categorical, CATEGORICAL_MODEL_PATH, ENCODER_PATH
from Classes.CRNN_class import CRNN

# Определяем путь к тестовому аудиофайлу. 
# Важно: замените этот путь на путь к реальному аудиофайлу для тестирования.
# Например, можно использовать один из ваших тестовых файлов в корне проекта.
TEST_AUDIO_PATH = os.path.join(project_root, "test.mp3") # Пример: используем test.mp3 из корня проекта

# Проверка существования тестового файла
if not os.path.exists(TEST_AUDIO_PATH):
    print(f"Ошибка: Тестовый аудиофайл не найден по пути: {TEST_AUDIO_PATH}")
    print("Пожалуйста, убедитесь, что файл существует или обновите путь TEST_AUDIO_PATH.")
    sys.exit(1)

def test_categorical_mood_model():
    """
    Тест для загрузки и использования модели категориальной классификации настроения.
    Примечание: Этот тест только проверяет загрузку и запуск инференса, 
    а не точность предсказаний, так как истинные метки настроения не доступны.
    """
    print("\n--- Запуск теста категориальной модели настроения ---")
    
    # Определение устройства (CPU/GPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {DEVICE}")

    # Загрузка Mood Encoder
    mood_encoder = None
    try:
        # Путь к файлу энкодера относительно этого тестового скрипта
        encoder_full_path = os.path.join(project_root, "Models", "mood_label_encoder_classification.pkl")
        if not os.path.exists(encoder_full_path):
            print(f"Ошибка: Файл Mood Encoder не найден по пути: {encoder_full_path}")
            return
        with open(encoder_full_path, 'rb') as f:
            mood_encoder = pickle.load(f)
        n_mood_classes = len(mood_encoder.classes_)
        print(f"Mood Encoder загружен успешно. Количество классов: {n_mood_classes}")
    except Exception as e:
        print(f"Ошибка при загрузке Mood Encoder: {e}")
        return

    # Загрузка категориальной модели настроения
    model_mood_categorical = None
    if mood_encoder is not None and n_mood_classes > 0:
        try:
            # Путь к файлу модели относительно этого тестового скрипта
            model_full_path = os.path.join(project_root, "Models", "crnn_deam_mood_classification.pth")
            if not os.path.exists(model_full_path):
                 print(f"Ошибка: Файл категориальной модели настроения не найден по пути: {model_full_path}")
                 return
            model_mood_categorical = CRNN(n_mels=128, n_classes=n_mood_classes).to(DEVICE)
            model_mood_categorical.load_state_dict(torch.load(model_full_path, map_location=DEVICE))
            model_mood_categorical.eval()
            print("Категориальная модель настроения загружена успешно.")
        except Exception as e:
            print(f"Ошибка при загрузке категориальной модели настроения: {e}")
            model_mood_categorical = None # Убедимся, что модель None при ошибке
            return
    else:
        print("Пропуск загрузки модели настроения, так как Mood Encoder не был загружен.")
        return

    # Проведение инференса для тестового файла
    if model_mood_categorical and mood_encoder:
        try:
            print(f"Проведение инференса для файла: {TEST_AUDIO_PATH}")
            predicted_mood = infer_mood_categorical(TEST_AUDIO_PATH, model_mood_categorical, mood_encoder, DEVICE)
            print(f"Предсказанная категория настроения: {predicted_mood}")
        except Exception as e:
            print(f"Ошибка при выполнении инференса: {e}")
    else:
        print("Пропуск инференса, так как модель или энкодер не загружены.")
        
    print("--- Тест категориальной модели настроения завершен ---")

# Запуск теста
if __name__ == '__main__':
    test_categorical_mood_model() 