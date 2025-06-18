import pickle
import uuid
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from fastapi.responses import JSONResponse
import shutil
import librosa
from yandex_music import Client
import json
import urllib.parse
import torch
from recognizers.instrument_detector import InstrumentRecognizer
from contextlib import asynccontextmanager


from Classes.CRNN_class import CRNN
from Classes.FMA_dataset import FMADataset, FMAMultiLabelDataset
from recognizers.AudD import recognize_song_with_acrcloud, recognize_song_with_audd
from recognizers.bpm_analyzer import estimate_bpm, estimate_bpm_advanced
from recognizers.genre_classifier import predict_multiclass, BASE_DIR
from recognizers.deam_infer import infer_mood_categorical, CATEGORICAL_MODEL_PATH, ENCODER_PATH



# Инициализация логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

with open(os.path.join(BASE_DIR, "../Models/genre_label_encoder_multiclass.pkl"), "rb") as f:
    le = pickle.load(f)

# Загрузка модели мультикласса
model_multiclass = CRNN(n_mels=128, n_classes=len(le.classes_))
model_multiclass.load_state_dict(torch.load(os.path.join(BASE_DIR, "../Models/crnn_fma_multiclass.pth"), map_location=DEVICE))
model_multiclass.to(DEVICE)
model_multiclass.eval()

# Добавить константы после BASE_DIR
MOOD_MODEL_PATH = os.path.join(BASE_DIR, "../Models/crnn_deam_mood_classification.pth")
MOOD_ENCODER_PATH = os.path.join(BASE_DIR, "../Models/mood_label_encoder_classification.pkl")

# Заменить существующие константы
CATEGORICAL_MODEL_PATH = MOOD_MODEL_PATH
ENCODER_PATH = MOOD_ENCODER_PATH

# --- Start: Load Categorical Mood Model and Encoder --- #
# Determine the number of classes for the mood model by loading the encoder
try:
    with open(ENCODER_PATH, 'rb') as f:
        mood_encoder = pickle.load(f)
    n_mood_classes = len(mood_encoder.classes_)
except FileNotFoundError:
    logger.error(f"Error: Mood encoder file not found at {ENCODER_PATH}. Cannot load mood model.")
    mood_encoder = None # Set to None if loading fails
    n_mood_classes = 0 # Set classes to 0 if loading fails
except Exception as e:
    logger.error(f"Error loading mood encoder: {e}")
    mood_encoder = None # Set to None if loading fails
    n_mood_classes = 0 # Set classes to 0 if loading fails

# Load the categorical mood model
model_mood_categorical = None # Initialize model as None
if n_mood_classes > 0: # Only attempt to load if encoder was loaded successfully
    try:
        model_mood_categorical = CRNN(n_mels=128, n_classes=n_mood_classes).to(DEVICE)
        model_mood_categorical.load_state_dict(torch.load(CATEGORICAL_MODEL_PATH, map_location=DEVICE))
        model_mood_categorical.eval()
        logger.info("Categorical mood model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: Categorical mood model file not found at {CATEGORICAL_MODEL_PATH}.")
        model_mood_categorical = None # Set to None if loading fails
    except Exception as e:
        logger.error(f"Error loading categorical mood model: {e}")
        model_mood_categorical = None # Set to None if loading fails
# --- End: Load Categorical Mood Model and Encoder --- #

# Добавить после других импортов, перед @asynccontextmanager
print("Проверяю модель инструментов при импорте...")

# Попробуем загрузить модель сразу
instrument_recognizer = None

def init_instrument_model():
    global instrument_recognizer
    try:
        from recognizers.instrument_detector import InstrumentRecognizer
        instrument_recognizer = InstrumentRecognizer("Models/crnn_irmas_instruments.pth")
        return True
    except Exception as e:
        return False

# Вызываем инициализацию сразу
init_success = init_instrument_model()

if init_success:
    print("Все модели успешно загружены.")
else:
    print("Ошибка загрузки одной или нескольких моделей.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=== ЗАПУСК СЕРВЕРА ===")
    # print(f"При запуске instrument_recognizer is None: {instrument_recognizer is None}")
    try:
        logger.info("Модели успешно загружены")
        yield
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
        import traceback
        traceback.print_exc()
        yield
    finally:
        # Shutdown
        print(" === ВЫКЛЮЧЕНИЕ СЕРВЕРА ===")

class UnicodeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

app = FastAPI(
    title="MusicNow API",
    description="API для анализа музыки",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=UnicodeJSONResponse
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Защита ключей через переменные окружения
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN", "61e9068ba431fc91c9596b86aebc5f5d")
LASTFM_TOKEN = os.getenv("LASTFM_TOKEN", "c1107c7c0f1cf4f9b31f7d4b84a98997")

# Данные для ACRCloud
ACR_HOST = "identify-eu-west-1.acrcloud.com"
ACR_ACCESS_KEY = "1d61906ff41dd262fc3ad008b5031fa4"
ACR_ACCESS_SECRET = "WbxvEOXRlM1Urx48y27u7Lob52LnPmKMwXrJFtlH"

# Добавить после создания других recognizers

def is_silent(file_path, silence_threshold=0.01, min_duration=1.0):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            return True  # Пустой файл
        duration = len(y) / sr
        if duration < min_duration:
            return True  # Слишком короткий файл
        rms = np.sqrt(np.mean(y**2))
        if rms < silence_threshold:
            return True  # Тишина
        return False
    except Exception as e:
        print(f"Ошибка при анализе тишины: {e}")
        return True  # Если не удалось прочитать файл — считаем его неподходящим


@app.post("/recognize", response_model=Dict[str, Any])
async def recognize_song(file: UploadFile = File(...)):
    """Анализ аудиофайла с гарантией сохранения файла."""
    temp_audio = None
    try:
        print(f"\n[DEBUG] Получен файл: {file.filename}")
        logger.info(f"Получен файл: {file.filename}")
        
        # Проверка расширения файла
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            print(f"[DEBUG] Неподдерживаемый формат файла: {file.filename}")
            logger.warning(f"Неподдерживаемый формат файла: {file.filename}")
            raise HTTPException(400, "Допустимы только MP3/WAV/M4A файлы!")

        # Создаем папку uploads, если её нет
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        print(f"[DEBUG] Директория для загрузки: {UPLOAD_DIR}")
        
        # Генерируем уникальное имя файла
        file_ext = Path(file.filename).suffix
        temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
        temp_audio = UPLOAD_DIR / temp_filename
        print(f"[DEBUG] Сохраняю как: {temp_audio}")

        # Сохраняем файл
        with open(temp_audio, 'wb') as f:
            content = await file.read()
            if not content:
                print("[DEBUG] Получен пустой файл")
                raise HTTPException(400, "Файл пуст!")
            f.write(content)
            print(f"[DEBUG] Файл сохранен ({len(content)} байт)")

        # Проверка на тишину
        if is_silent(temp_audio):
            print("[DEBUG] Файл содержит только тишину!")
            raise HTTPException(400, "Файл содержит только тишину!")

        print("[DEBUG] Начинаю анализ аудио...")
        metadata = await analyze_audio(temp_audio)
        print("[DEBUG] Анализ метаданных завершен")
        
        print("[DEBUG] Начинаю распознавание песни...")
        audd_result = await recognize_with_audd(temp_audio, use_audd=True)
        print("[DEBUG] Распознавание завершено")

        # Поиск ссылки на Яндекс Музыке
        yandex_music_url = None
        youtube_music_url = None
        if audd_result and audd_result.get('artist') and audd_result.get('title'):
            yandex_music_url = get_yandex_music_link(audd_result['artist'], audd_result['title'])
            youtube_music_url = get_youtube_music_link(audd_result['artist'], audd_result['title'])

        # Формируем ответ
        response = {
            "metadata": metadata,
            "audd": audd_result,
            "yandex_music_url": yandex_music_url,
            "youtube_music_url": youtube_music_url
        }
        log_file = UPLOAD_DIR / "last_response.json"
        with open(log_file, "w") as f:
            json.dump(response, f, indent=2)
        logger.info(f"Финальный ответ сохранён в {log_file}")
        
        print("\n[DEBUG] === Подготовленный ответ ===")
        print(f"[DEBUG] Метаданные:")
        print(f"[DEBUG] - Жанр: {metadata.get('genre')}")
        print(f"[DEBUG] - BPM: {metadata.get('bpm')}")
        print(f"[DEBUG] - Уверенность: {metadata.get('confidence')}")
        print(f"[DEBUG] - Инструменты: {metadata.get('instruments')}")
        print(f"[DEBUG] - Настроение: {metadata.get('mood_category')}")
        
        if audd_result:
            print(f"[DEBUG] Данные распознавания:")
            print(f"[DEBUG] - Исполнитель: {audd_result.get('artist')}")
            print(f"[DEBUG] - Название: {audd_result.get('title')}")
            print(f"[DEBUG] - Альбом: {audd_result.get('album')}")
            print(f"[DEBUG] - Дата релиза: {audd_result.get('release_date')}")
        print(f"[DEBUG] Ссылка на Яндекс.Музыку: {yandex_music_url or 'Не найдена'}")
        print(f"[DEBUG] Ссылка на YouTube Music: {youtube_music_url or 'Не найдена'}")
        print("[DEBUG] ========================\n")

        try:
            shutil.copy(str(temp_audio), "uploads/last_uploaded_file.m4a")
            print("[DEBUG] Копия файла для отладки сохранена: uploads/last_uploaded_file.m4a")
            logger.info("Сделана копия файла для отладки: uploads/last_uploaded_file.m4a")
        except Exception as e:
            print(f"[DEBUG] Не удалось скопировать файл для отладки: {e}")
            logger.warning(f"Не удалось скопировать файл для отладки: {e}")

        return response

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Ошибка сервера")
    finally:
        if temp_audio and temp_audio.exists():
            try:
                os.remove(temp_audio)
                logger.info(f"Временный файл удален: {temp_audio}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл: {str(e)}")


async def analyze_audio(file_path: Path) -> Dict[str, Any]:
    try:
        logger.info("Файл получен: %s", file_path)
        logger.info("Начинаю анализ аудио")

        # Отладка инструментов
        print(f"Анализирую инструменты...")
        print(f"instrument_recognizer is None: {instrument_recognizer is None}")
        
        if instrument_recognizer:
            print(f"Вызываю instrument_recognizer.predict для {file_path}")
            try:
                instrument_result = instrument_recognizer.predict(str(file_path), top_k=5)
                print(f"Результат предсказания: {instrument_result}")
                instruments = [item['instrument'] for item in instrument_result]
                print(f"Список инструментов: {instruments}")
            except Exception as e:
                print(f"Ошибка в predict: {e}")
                import traceback
                traceback.print_exc()
                instruments = []
        else:
            print(f"instrument_recognizer не загружен!")
            instruments = []
            
        logger.info(f"Detected instruments: {instruments}")
        
        bpm_result = estimate_bpm_advanced(str(file_path))
        bpm = bpm_result.get("bpm", 0.0)
        bpm_confidence = bpm_result.get("confidence", 0.0)
        bpm_alternatives = bpm_result.get("alternative_bpm", [])
        rhythm_stability = bpm_result.get("rhythm_stability", 0.0)
        genre, confidence, _ = predict_multiclass(str(file_path), model_multiclass, le, DEVICE)
        logger.info("Вызов predict_multiclass с аргументами: %s, %s, %s, %s", file_path, model_multiclass, le, DEVICE)
        logger.info("Результат жанра: %s", genre)

        # Perform categorical mood inference
        predicted_mood_category = None # Initialize as None
        if model_mood_categorical and mood_encoder: # Only run if model and encoder are loaded
            logger.info("Performing categorical mood inference")
            predicted_mood_category = infer_mood_categorical(str(file_path), model_mood_categorical, mood_encoder, DEVICE)
            logger.info(f"Predicted mood category: {predicted_mood_category}")
        else:
             logger.warning("Categorical mood model or encoder not loaded. Skipping mood inference.")

        # Проверка на пустые значения (adjusted check for instruments - now checks the cleaned list)
        if not instruments or bpm is None or genre is None:
             # Allow mood to be None if model/encoder failed to load
            if predicted_mood_category is None:
                 if not instruments and bpm is None or genre is None:
                     raise ValueError("Не удалось распознать часть метаданных (BPM, жанр) и модель настроения недоступна.")
                 elif bpm is None or genre is None:
                      raise ValueError("Не удалось распознать часть метаданных (BPM, жанр) и модель настроения недоступна.")

            # If mood is available, still raise error for other missing metadata
            if not instruments and bpm is None or genre is None:
                 raise ValueError("Не удалось распознать часть метаданных (BPM, жанр)")
            elif bpm is None or genre is None:
                 raise ValueError("Не удалось распознать часть метаданных (BPM, жанр)")

        # Приведение к стандартным типам (instruments is now the cleaned list of strings)
        genre = str(genre)
        confidence = float(confidence) if confidence is not None else None
        if isinstance(bpm, np.generic):
            bpm = float(bpm)
        # instruments should already be a list of strings

        return {
            "instruments": instruments,
            "bpm": bpm,
            "bmp_confidence": bpm_confidence,
            "bpm_alternatives": bpm_alternatives,
            "rhythm_stability": rhythm_stability,
            "genre": genre,
            "confidence": confidence,
            "mood_category": predicted_mood_category
        }

    except Exception as e:
        logger.error("Ошибка: %s", str(e), exc_info=True)
        raise HTTPException(500, f"Анализ аудио не удался: {str(e)}")


async def recognize_with_audd(file_path: Path, use_audd: bool = True) -> dict:
    """Распознавание через AudD или ACRCloud"""
    try:
        logger.info("Начинаю распознавание песни...")
        if use_audd:
            logger.info("Использую AudD API")
            result = recognize_song_with_audd(str(file_path), AUDD_API_TOKEN)
        else:
            logger.info("Использую ACRCloud API")
            result = recognize_song_with_acrcloud(
                str(file_path),
                ACR_HOST,
                ACR_ACCESS_KEY,
                ACR_ACCESS_SECRET
            )
        logger.info("Распознавание завершено успешно")
        return result
    except Exception as e:
        logger.error(f"Audio recognition error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    return {
        "status": "OK",
        "models_loaded": True,
        "instrument_model_loaded": instrument_recognizer is not None,
        "instrument_model_path_exists": os.path.exists("Models/crnn_irmas_instruments.pth")
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )

def get_yandex_music_link(artist: str, title: str) -> str:
    try:
        client = Client().init()  # Без авторизации можно только искать
        search_text = f"{artist} {title}"
        search_result = client.search(search_text)
        if search_result.tracks and search_result.tracks.results:
            track = search_result.tracks.results[0]
            track_id = track.id
            album_id = track.albums[0].id
            return f"https://music.yandex.ru/album/{album_id}/track/{track_id}"
    except Exception as e:
        print(f"Yandex Music search error: {e}")
    return None

def get_youtube_music_link(artist: str, title: str) -> str:
    query = f"{artist} {title}"
    encoded_query = urllib.parse.quote_plus(query)
    return f"https://music.youtube.com/search?q={encoded_query}"

@app.post("/analyze_instruments")
async def analyze_instruments(file: UploadFile = File(...)):
    """Анализ инструментов в аудиофайле"""
    global instrument_recognizer
    
    if instrument_recognizer is None:
        raise HTTPException(503, "Instrument model not loaded yet")
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(400, "Допустимы только MP3/WAV/M4A файлы!")
    
    temp_audio = None
    try:
        file_ext = Path(file.filename).suffix
        temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
        temp_audio = UPLOAD_DIR / temp_filename
        
        with open(temp_audio, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        instruments = instrument_recognizer.predict(str(temp_audio), top_k=5)
        
        return {
            'instruments': instruments,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Instrument analysis error: {str(e)}")
        raise HTTPException(500, f"Ошибка анализа инструментов: {str(e)}")
    finally:
        if temp_audio and temp_audio.exists():
            os.remove(temp_audio)