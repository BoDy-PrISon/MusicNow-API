import os
import numpy as np
import torch
import librosa
import pickle
from Classes.CRNN_class import CRNN

# Получаем абсолютный путь к файлу модели относительно этого скрипта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORICAL_MODEL_PATH = os.path.join(BASE_DIR, '../Models/crnn_deam_mood_classification.pth')
ENCODER_PATH = os.path.join(BASE_DIR, '../Models/mood_label_encoder_classification.pkl')
AUDIO_DIR = r'C:/Users/Fin/PycharmProjects/MusicNow Api/DEAM/deam-mediaeval-dataset-emotional-analysis-in-music/versions/1/DEAM_audio/MEMD_audio'
SR = 22050
N_MELS = 128
SEGMENT_SECONDS = 20
STEP_SECONDS = 10

# Новая функция для категориальной классификации настроения
def infer_mood_categorical(audio_path, model, encoder, device):
    y, sr = librosa.load(audio_path, sr=SR)
    segment_samples = SEGMENT_SECONDS * SR
    step_samples = STEP_SECONDS * SR
    segments = []
    # Если аудио короче одного сегмента, дополняем нулями
    if len(y) < segment_samples:
        y_seg = np.pad(y, (0, segment_samples - len(y)))
        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        segments.append(mel_db)
    else:
        for start in range(0, len(y) - segment_samples + 1, step_samples):
            y_seg = y[start:start+segment_samples]
            mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            segments.append(mel_db)
        # Если остался хвост < segment_samples, можно проигнорировать или дополнить до сегмента
        if (len(y) % step_samples != 0) and (len(y) > segment_samples):
            last_start = len(y) - segment_samples
            y_seg = y[last_start:]
            mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            segments.append(mel_db)

    if not segments:
        return "Could not process audio segments."

    X = np.stack(segments)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, n_mels, time)

    model.eval()
    with torch.no_grad():
        # Get predictions for each segment
        segment_preds = model(X.to(device)).cpu().numpy()

    # Average the predictions (probabilities) across segments
    mean_preds = np.mean(segment_preds, axis=0)

    # Get the predicted class index
    predicted_class_index = np.argmax(mean_preds)

    # Load the encoder and decode the class index
    try:
        with open(ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        predicted_mood = le.inverse_transform([predicted_class_index])[0]
    except FileNotFoundError:
        return f"Error: Encoder file not found at {ENCODER_PATH}"
    except Exception as e:
        return f"Error loading or using encoder: {e}"

    return predicted_mood

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        with open(ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        n_classes = len(le.classes_)
    except FileNotFoundError:
        print(f"Error: Encoder file not found at {ENCODER_PATH}. Cannot determine number of classes.")
        return
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return

    model = CRNN(n_mels=N_MELS, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(CATEGORICAL_MODEL_PATH, map_location=device))
    model.eval()

    try:
        with open(ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Encoder file not found at {ENCODER_PATH}.")
        return
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return

    audio_file = input('Введите путь к аудиофайлу (или просто имя файла в корне репозитория): ').strip()
    if not os.path.isabs(audio_file) and not os.path.exists(audio_file):
        repo_root = os.path.abspath(os.path.join(BASE_DIR, '..'))
        candidate = os.path.join(repo_root, audio_file)
        if os.path.exists(candidate):
            audio_file = candidate
        else:
            candidate = os.path.join(AUDIO_DIR, audio_file)
            if os.path.exists(candidate):
                audio_file = candidate
    if not os.path.exists(audio_file):
        print(f'Файл {audio_file} не найден!')
    else:
        result = infer_mood_categorical(audio_file, model, le, device)
        print(f"Predicted Mood: {result}")

if __name__ == '__main__':
    main() 