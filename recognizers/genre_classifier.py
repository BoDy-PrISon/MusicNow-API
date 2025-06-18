import os
import pickle
import torch
import librosa
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path

from Classes.CRNN_class import CRNN

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenreClassifier:
    """
    Классификатор музыкальных жанров (multiclass)
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Args:
            base_dir: Базовая директория для моделей
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy loading - модель загружается при первом использовании
        self.model = None
        self.encoder = None
        self.classes = None
        
        logger.info(f"GenreClassifier initialized on {self.device}")
    
    def _get_model_paths(self) -> Dict[str, Path]:
        """Получить пути к файлам моделей"""
        models_dir = Path(self.base_dir) / ".." / "Models"
        
        return {
            'model': models_dir / "crnn_fma_multiclass.pth",
            'encoder': models_dir / "genre_label_encoder_multiclass.pkl"
        }
    
    def _validate_model_files(self) -> None:
        """Проверить существование файлов моделей"""
        paths = self._get_model_paths()
        
        for file_type, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{file_type.title()} file not found: {path}")
    
    def _load_model(self) -> None:
        """Загрузить модель и энкодер"""
        if self.model is not None:  # Уже загружено
            return
        
        try:
            self._validate_model_files()
            paths = self._get_model_paths()
            
            # Загрузка энкодера
            with open(paths['encoder'], 'rb') as f:
                self.encoder = pickle.load(f)
                self.classes = self.encoder.classes_
            
            # Загрузка модели
            self.model = CRNN(n_mels=128, n_classes=len(self.classes))
            self.model.load_state_dict(torch.load(paths['model'], map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully. Classes: {len(self.classes)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _preprocess_audio(self, audio_path: str, n_mels: int = 128, 
                         duration: int = 30, sr: int = 22050) -> torch.Tensor:
        """
        Предобработка аудиофайла в мел-спектрограмму
        
        Args:
            audio_path: Путь к аудиофайлу
            n_mels: Количество мел-фильтров
            duration: Длительность в секундах
            sr: Частота дискретизации
            
        Returns:
            Тензор мел-спектрограммы
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Проверка размера файла (макс 100MB)
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        if file_size > 100:
            raise ValueError(f"File too large: {file_size:.1f}MB (max 100MB)")
        
        try:
            y, loaded_sr = librosa.load(audio_path, sr=sr, duration=duration)
            samples = sr * duration
            
            if len(y) < samples:
                y = np.pad(y, (0, samples - len(y)))
            else:
                # Случайная обрезка для разнообразия
                if len(y) > samples:
                    start = np.random.randint(0, len(y) - samples)
                    y = y[start:start + samples]
            
            # Создание мел-спектрограммы
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Нормализация
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            
            return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def predict(self, audio_path: str, top_k: int = 3) -> Tuple[str, float, Dict[str, float]]:
        """
        Предсказание жанра
        
        Args:
            audio_path: Путь к аудиофайлу
            top_k: Количество топ предсказаний
            
        Returns:
            (predicted_genre, confidence, top_k_predictions)
        """
        self._load_model()
        
        try:
            # Предобработка
            mel_tensor = self._preprocess_audio(audio_path).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                logits = self.model(mel_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Топ-K предсказаний
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_predictions = {
                self.encoder.inverse_transform([idx])[0]: float(probs[idx])
                for idx in top_indices
            }
            
            # Основное предсказание
            pred_idx = probs.argmax()
            predicted_genre = self.encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            logger.info(f"Predicted genre: {predicted_genre} (confidence: {confidence:.3f})")
            return predicted_genre, confidence, top_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, any]:
        """Получить информацию о модели"""
        self._load_model()
        
        return {
            'device': str(self.device),
            'num_classes': len(self.classes),
            'classes': list(self.classes),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }


# Глобальный экземпляр для удобства
_classifier = None

def get_classifier() -> GenreClassifier:
    """Получить глобальный экземпляр классификатора"""
    global _classifier
    if _classifier is None:
        _classifier = GenreClassifier()
    return _classifier

# Функция для обратной совместимости с main.py
def predict_multiclass(audio_path: str, model=None, le=None, device=None, **kwargs) -> Tuple[str, float, np.ndarray]:
    """
    Обратная совместимость с main.py
    
    Returns:
        (genre, confidence, probabilities_array)
    """
    classifier = get_classifier()
    genre, confidence, top_predictions = classifier.predict(audio_path)
    
    # Создаем массив вероятностей в том же порядке, что и классы энкодера
    probs = np.zeros(len(classifier.classes))
    for i, class_name in enumerate(classifier.classes):
        if class_name in top_predictions:
            probs[i] = top_predictions[class_name]
        else:
            # Для классов не в top_k нужно вычислить вероятности
            # Загружаем модель и делаем полное предсказание
            mel_tensor = classifier._preprocess_audio(audio_path).to(classifier.device)
            with torch.no_grad():
                logits = classifier.model(mel_tensor)
                full_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs = full_probs
            break
    
    return genre, confidence, probs

# Экспорт BASE_DIR для совместимости
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пример использования
if __name__ == "__main__":
    classifier = GenreClassifier()
    
    audio_path = "../test2.mp3"
    if os.path.exists(audio_path):
        try:
            genre, confidence, top_predictions = classifier.predict(audio_path)
            print(f"Predicted genre: {genre} (confidence: {confidence:.3f})")
            print("Top predictions:")
            for g, prob in top_predictions.items():
                print(f"  {g}: {prob:.3f}")
            
            # Информация о модели
            info = classifier.get_model_info()
            print(f"\nModel info:")
            print(f"  Device: {info['device']}")
            print(f"  Classes: {info['num_classes']}")
            print(f"  Parameters: {info['model_parameters']:,}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"File not found: {audio_path}")