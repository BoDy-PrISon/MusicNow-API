import tempfile
import librosa
import soundfile as sf
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def estimate_bpm(audio_path: str) -> dict:
    try:
        # Принудительно конвертируем в WAV если нужно
        if not audio_path.lower().endswith('.wav'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                logger.info(f"🔄 Конвертируем в WAV: {audio_path} -> {tmp_wav.name}")
                y, sr = librosa.load(audio_path, sr=None)
                sf.write(tmp_wav.name, y, sr)
                audio_path = tmp_wav.name

        logger.info(f"📀 Загружаем аудиофайл: {audio_path}")
        y, sr = librosa.load(audio_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo), 2)
        logger.info(f"🎶 BPM: {bpm}")
        return {"bpm": bpm}

    except Exception as e:
        logger.exception(f"BPM Error: {str(e)}")
        return {"bpm": 0.0}

def estimate_bpm_advanced(audio_path: str) -> dict:
    try:
        # Принудительно конвертируем в WAV если нужно
        if not audio_path.lower().endswith('.wav'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                logger.info(f"🔄 Конвертируем в WAV: {audio_path} -> {tmp_wav.name}")
                y, sr = librosa.load(audio_path, sr=None)
                sf.write(tmp_wav.name, y, sr)
                audio_path = tmp_wav.name

        logger.info(f"📀 Загружаем аудиофайл: {audio_path}")
        y, sr = librosa.load(audio_path)
        
        # Основной анализ темпа
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Анализ onset envelope для confidence
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Расчёт уверенности
        confidence = calculate_rhythm_confidence(onset_env, beats, sr)
        
        # Альтернативные BPM (для сложных ритмов)
        alternative_bpm = [
            round(float(tempo * 2), 2),  # Двойной темп
            round(float(tempo / 2), 2),  # Половинный темп
        ]
        
        # Анализ стабильности ритма
        rhythm_stability = calculate_rhythm_stability(beats)
        
        bpm = round(float(tempo), 2)
        logger.info(f"🎶 BPM: {bpm} (confidence: {confidence:.2f})")
        
        return {
            "bpm": bpm,
            "confidence": confidence,
            "alternative_bpm": alternative_bpm,
            "rhythm_stability": rhythm_stability,
            "beat_count": len(beats)
        }

    except Exception as e:
        logger.exception(f"BPM Error: {str(e)}")
        return {
            "bpm": 0.0,
            "confidence": 0.0,
            "alternative_bpm": [],
            "rhythm_stability": 0.0,
            "beat_count": 0
        }

def calculate_rhythm_confidence(onset_env, beats, sr):
    """
    Рассчитывает уверенность в определении ритма
    """
    try:
        # Нормализуем onset envelope
        onset_env_norm = onset_env / (np.max(onset_env) + 1e-6)
        
        # Стандартное отклонение - чем выше, тем ритм выраженнее
        std_dev = np.std(onset_env_norm)
        
        # Энергия пиков в местах битов
        beat_frames = librosa.frames_to_samples(beats, sr=sr)
        beat_frames = beat_frames[beat_frames < len(onset_env)]
        
        if len(beat_frames) > 0:
            beat_energy = np.mean(onset_env_norm[beat_frames])
            # Сравниваем с общей энергией
            avg_energy = np.mean(onset_env_norm)
            energy_ratio = beat_energy / (avg_energy + 1e-6)
        else:
            energy_ratio = 0.0
        
        # Комбинируем метрики
        confidence = min(1.0, (std_dev * 2 + energy_ratio) / 3)
        return round(confidence, 3)
        
    except Exception:
        return 0.0

def calculate_rhythm_stability(beats):
    """
    Рассчитывает стабильность ритма (насколько равномерны интервалы между битами)
    """
    try:
        if len(beats) < 3:
            return 0.0
            
        # Интервалы между битами
        intervals = np.diff(beats)
        
        # Коэффициент вариации (CV) - чем меньше, тем стабильнее ритм
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval > 0:
            cv = std_interval / mean_interval
            # Преобразуем в стабильность (1 - нормализованный CV)
            stability = max(0.0, 1.0 - min(1.0, cv * 2))
        else:
            stability = 0.0
            
        return round(stability, 3)
        
    except Exception:
        return 0.0



