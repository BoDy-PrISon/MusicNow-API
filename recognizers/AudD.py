import requests
import time
import hashlib
import hmac
import base64
import os
from pydub import AudioSegment
import tempfile
import json

# Функция для распознавания трека через ACRCloud

def recognize_song_with_acrcloud(file_path: str, host: str, access_key: str, access_secret: str) -> dict:
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(int(time.time()))

    string_to_sign = "\n".join([http_method, http_uri, access_key, data_type, signature_version, timestamp])
    sign = base64.b64encode(hmac.new(access_secret.encode('utf-8'), string_to_sign.encode('utf-8'), digestmod=hashlib.sha1).digest()).decode('utf-8')

    # --- Обрезка аудио до 15 секунд ---
    temp_file = None
    try:
        audio = AudioSegment.from_file(file_path)
        if len(audio) > 20 * 1000:  # если длиннее 20 секунд
            audio = audio[:15 * 1000]  # обрезаем до 15 секунд
            temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
            audio.export(temp_path, format="mp3")
            os.close(temp_fd)
            file_to_send = temp_path
            temp_file = temp_path
        else:
            file_to_send = file_path

        with open(file_to_send, 'rb') as f:
            files = {'sample': f}
            data = {
                'access_key': access_key,
                'data_type': data_type,
                'signature_version': signature_version,
                'signature': sign,
                'timestamp': timestamp,
            }
            url = f'https://{host}/v1/identify'
            response = requests.post(url, files=files, data=data, timeout=10)
            result = response.json()
            if result.get('status', {}).get('msg') == 'Success':
                metadata = result.get('metadata', {})
                music = metadata.get('music', [{}])[0]
                return {
                    'title': music.get('title'),
                    'artist': ', '.join([a['name'] for a in music.get('artists', [])]) if music.get('artists') else None,
                    'album': music.get('album', {}).get('name'),
                    'acrid': music.get('acrid'),
                    'score': music.get('score'),
                    'label': music.get('label'),
                    'result_type': music.get('result_type'),
                    'external_metadata': music.get('external_metadata', {})
                }
            else:
                return {'status': 'error', 'message': result.get('status', {}).get('msg', 'Unknown error')}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    finally:
        # Удаляем временный файл, если он был создан
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

def recognize_song_with_audd(file_path: str, api_token: str) -> dict:
    temp_file = None
    try:
        # Загружаем и конвертируем файл в MP3
        print(f"[DEBUG] Загрузка файла: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        # Обрезаем до 15 секунд, если нужно
        if len(audio) > 20 * 1000:
            print("[DEBUG] Обрезка аудио до 15 секунд")
            audio = audio[:15 * 1000]
        
        # Создаем временный файл MP3
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)
        print(f"[DEBUG] Конвертация в MP3: {temp_path}")
        
        # Экспортируем в MP3 с хорошим качеством
        audio.export(
            temp_path,
            format="mp3",
            bitrate="192k",
            parameters=["-q:a", "0"]  # Высокое качество audio
        )
        temp_file = temp_path
        print("[DEBUG] Конвертация завершена")

        # Отправляем запрос
        print("[DEBUG] Отправка запроса в AudD")
        with open(temp_file, 'rb') as f:
            files = {'file': f}
            data = {
                'api_token': api_token,
                'return': 'apple_music,spotify',
            }
            url = 'https://api.audd.io/'
            response = requests.post(url, data=data, files=files, timeout=15)
            result = response.json()
            print(f"[DEBUG] Полный ответ от AudD: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('status') == 'success' and result.get('result'):
                print("[DEBUG] Успешное распознавание")
                return result['result']
            else:
                print(f"[DEBUG] Ошибка распознавания: status={result.get('status')}, error={result.get('error')}")
                return {'status': 'error', 'message': result.get('error', 'Unknown error')}
    except Exception as e:
        print(f"[DEBUG] Ошибка при обработке: {str(e)}")
        return {'status': 'error', 'message': str(e)}
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"[DEBUG] Удален временный файл: {temp_file}")
            except Exception as e:
                print(f"[DEBUG] Ошибка при удалении временного файла: {str(e)}")




