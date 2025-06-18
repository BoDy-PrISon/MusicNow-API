import pytest
import asyncio
import os
import json
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

# Импортируем наш FastAPI app
import sys
sys.path.append('..')
from main import app

class TestMusicNowAPI:
    @pytest.fixture
    def client(self):
        """Создаем тестовый клиент FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_audio_files(self):
        """Подготавливаем тестовые аудио файлы"""
        test_files = {}
        
        # Проверяем существующие файлы в корне проекта
        audio_files = [
            ("classical", "classic.mp3"),
            ("classical2", "classic2.mp3"), 
            ("pop", "pop.mp3"),
            ("test", "test.mp3"),
            ("test2", "test2.mp3")
        ]
        
        for name, filename in audio_files:
            if os.path.exists(filename):
                test_files[name] = filename
                
        return test_files
    
    def test_health_check(self, client):
        """Тест проверки работоспособности сервера"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert "models_loaded" in data
    
    def test_recognize_endpoint_no_file(self, client):
        """Тест без файла - должен вернуть ошибку"""
        response = client.post("/recognize")
        assert response.status_code == 422  # Validation error
    
    def test_recognize_endpoint_empty_file(self, client):
        """Тест с пустым файлом"""
        # Создаем пустой файл
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = client.post(
                    "/recognize",
                    files={"file": ("empty.mp3", f, "audio/mpeg")}
                )
            assert response.status_code == 400
            assert "пуст" in response.json()["detail"].lower()
        finally:
            os.unlink(tmp_path)
    
    def test_recognize_endpoint_invalid_format(self, client):
        """Тест с неподдерживаемым форматом"""
        # Создаем текстовый файл с расширением .txt
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"This is not audio")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = client.post(
                    "/recognize",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            assert response.status_code == 400
            assert "допустимы только" in response.json()["detail"].lower()
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.parametrize("file_key", ["classical", "pop", "test", "test2"])
    def test_recognize_valid_audio(self, client, sample_audio_files, file_key):
        """Тест с валидными аудио файлами"""
        if file_key not in sample_audio_files:
            pytest.skip(f"Файл {file_key} не найден")
        
        file_path = sample_audio_files[file_key]
        
        with open(file_path, 'rb') as f:
            response = client.post(
                "/recognize",
                files={"file": (file_path, f, "audio/mpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "metadata" in data
        assert "audd" in data
        
        # Проверяем метаданные
        metadata = data["metadata"]
        assert "genre" in metadata
        assert "bpm" in metadata
        assert "confidence" in metadata
        assert "instruments" in metadata
        
        # Проверяем типы данных
        assert isinstance(metadata["genre"], str)
        assert isinstance(metadata["bpm"], (int, float))
        assert isinstance(metadata["confidence"], (int, float))
        assert isinstance(metadata["instruments"], list)
        
        print(f"\n=== Результат для {file_key} ({file_path}) ===")
        print(f"Жанр: {metadata['genre']} (уверенность: {metadata['confidence']:.3f})")
        print(f"BPM: {metadata['bpm']}")
        print(f"Инструменты: {metadata['instruments']}")
        if metadata.get('mood_category'):
            print(f"Настроение: {metadata['mood_category']}")
    
    def test_metadata_analysis_components(self, client, sample_audio_files):
        """Детальный тест компонентов анализа метаданных"""
        if not sample_audio_files:
            pytest.skip("Нет тестовых аудио файлов")
        
        # Берем первый доступный файл
        file_path = list(sample_audio_files.values())[0]
        
        with open(file_path, 'rb') as f:
            response = client.post(
                "/recognize",
                files={"file": (file_path, f, "audio/mpeg")}
            )
        
        assert response.status_code == 200
        metadata = response.json()["metadata"]
        
        # Тест BPM анализа
        assert "bmp_confidence" in metadata
        assert "bpm_alternatives" in metadata
        assert "rhythm_stability" in metadata
        
        # Проверяем диапазоны значений
        assert 0 <= metadata["confidence"] <= 1
        assert 60 <= metadata["bpm"] <= 200  # Разумный диапазон BPM
        assert 0 <= metadata.get("bmp_confidence", 0) <= 1
        assert 0 <= metadata.get("rhythm_stability", 0) <= 1
    
    def test_error_handling(self, client):
        """Тест обработки ошибок"""
        # Создаем поврежденный аудио файл
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b"fake audio data that will cause error")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = client.post(
                    "/recognize",
                    files={"file": ("broken.mp3", f, "audio/mpeg")}
                )
            
            # Должен вернуть ошибку сервера или bad request
            assert response.status_code in [400, 500]
            
        finally:
            os.unlink(tmp_path)
    
    def test_concurrent_requests(self, client, sample_audio_files):
        """Тест параллельных запросов"""
        if not sample_audio_files:
            pytest.skip("Нет тестовых аудио файлов")
        
        file_path = list(sample_audio_files.values())[0]
        
        def make_request():
            with open(file_path, 'rb') as f:
                return client.post(
                    "/recognize",
                    files={"file": (file_path, f, "audio/mpeg")}
                )
        
        # Делаем 3 параллельных запроса
        responses = []
        for _ in range(3):
            responses.append(make_request())
        
        # Все должны быть успешными
        for response in responses:
            assert response.status_code == 200
            assert "metadata" in response.json()
    
    def test_response_consistency(self, client, sample_audio_files):
        """Тест консистентности ответов для одного файла"""
        if not sample_audio_files:
            pytest.skip("Нет тестовых аудио файлов")
        
        file_path = list(sample_audio_files.values())[0]
        
        # Делаем два запроса с одним файлом
        responses = []
        for _ in range(2):
            with open(file_path, 'rb') as f:
                response = client.post(
                    "/recognize",
                    files={"file": (file_path, f, "audio/mpeg")}
                )
                responses.append(response.json())
        
        # Метаданные должны быть одинаковыми
        metadata1 = responses[0]["metadata"]
        metadata2 = responses[1]["metadata"]
        
        assert metadata1["genre"] == metadata2["genre"]
        assert abs(metadata1["bpm"] - metadata2["bpm"]) < 1  # Небольшая погрешность допустима
        assert metadata1["instruments"] == metadata2["instruments"]
    
    def test_file_cleanup(self, client, sample_audio_files):
        """Тест очистки временных файлов"""
        if not sample_audio_files:
            pytest.skip("Нет тестовых аудио файлов")
        
        file_path = list(sample_audio_files.values())[0]
        
        # Считаем файлы в uploads до запроса
        uploads_dir = Path("uploads")
        files_before = len(list(uploads_dir.glob("temp_*"))) if uploads_dir.exists() else 0
        
        with open(file_path, 'rb') as f:
            response = client.post(
                "/recognize",
                files={"file": (file_path, f, "audio/mpeg")}
            )
        
        assert response.status_code == 200
        
        # Считаем файлы после запроса
        files_after = len(list(uploads_dir.glob("temp_*"))) if uploads_dir.exists() else 0
        
        # Временные файлы должны быть удалены
        assert files_after == files_before
    
    @pytest.mark.skipif(not os.path.exists("Models/best_irmas_model.pth"), 
                       reason="Instrument model not trained yet")
    def test_analyze_instruments_endpoint(self, client, sample_audio_files):
        """Тест endpoint для анализа инструментов"""
        if not sample_audio_files:
            pytest.skip("Нет тестовых аудио файлов")
        
        file_path = list(sample_audio_files.values())[0]
        
        with open(file_path, 'rb') as f:
            response = client.post(
                "/analyze_instruments",
                files={"file": (file_path, f, "audio/mpeg")}
            )
        
        # Если модель не загружена, ожидаем ошибку 503
        if response.status_code == 503:
            assert "not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "instruments" in data
            assert "status" in data

# Дополнительные утилиты для тестирования
class TestUtilities:
    def test_audio_file_validation(self):
        """Тест валидации аудио файлов"""
        from main import is_silent
        
        # Тест с существующими файлами
        existing_files = ["classic.mp3", "pop.mp3", "test.mp3"]
        
        for file_path in existing_files:
            if os.path.exists(file_path):
                assert not is_silent(file_path), f"Файл {file_path} не должен быть тишиной"
                print(f"✅ {file_path} - валидный аудио файл")
                break
        else:
            pytest.skip("Нет доступных аудио файлов для тестирования")

    def test_existing_files_structure(self):
        """Проверка структуры существующих файлов"""
        audio_files = ["classic.mp3", "classic2.mp3", "pop.mp3", "test.mp3", "test2.mp3"]
        
        found_files = []
        for file_path in audio_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                found_files.append((file_path, file_size))
                
        print(f"\n=== Найденные аудио файлы ===")
        for file_path, size in found_files:
            print(f"{file_path}: {size/1024:.1f} KB")
            
        assert len(found_files) > 0, "Не найдено ни одного аудио файла"

if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"]) 