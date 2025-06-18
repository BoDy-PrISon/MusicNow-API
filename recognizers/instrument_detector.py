import torch
import librosa
import numpy as np
from Classes.CRNN_class import CRNN

class InstrumentRecognizer:
    def __init__(self, model_path="Models/crnn_irmas_instruments.pth"):
        print(f"Инициализирую InstrumentRecognizer с {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {self.device}")
        
        self.model = CRNN(n_mels=128, n_classes=11)
        print(f"Модель создана")
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Веса модели загружены")
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        print(f"Модель готова к работе")
        
        # Английские сокращения
        self.class_names = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        
        # Русские полные названия
        self.russian_names = {
            'cel': 'Виолончель',
            'cla': 'Кларнет', 
            'flu': 'Флейта',
            'gac': 'Акустическая гитара',
            'gel': 'Электрогитара',
            'org': 'Орган',
            'pia': 'Фортепиано',
            'sax': 'Саксофон',
            'tru': 'Труба',
            'vio': 'Скрипка',
            'voi': 'Вокал'
        }
        
        print(f"Классы инструментов: {self.class_names}")
        
    def predict(self, audio_path, top_k=3):
        """Предсказать инструменты в аудио файле"""
        # Загрузить и обработать аудио
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        # Создать мел-спектрограмму
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Нормализация
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        
        # Подготовить для модели
        input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Получить топ-k результатов
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            eng_name = self.class_names[idx]
            rus_name = self.russian_names[eng_name]
            results.append({
                'instrument': rus_name,  # Теперь русское название
                'confidence': float(probabilities[idx])
            })
            
        return results
