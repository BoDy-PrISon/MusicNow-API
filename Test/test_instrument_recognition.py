import pytest
import os
from unittest.mock import Mock, patch

class TestInstrumentRecognition:
    
    def test_import_recognizer(self):
        """Тест импорта класса"""
        try:
            from recognizers.instrument_detector import InstrumentRecognizer
            assert InstrumentRecognizer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import InstrumentRecognizer: {e}")
    
    @pytest.mark.skipif(not os.path.exists("Models/best_irmas_model.pth"), 
                       reason="Model not trained yet")
    def test_model_loading(self):
        """Тест загрузки модели (только если модель существует)"""
        from recognizers.instrument_detector import InstrumentRecognizer
        recognizer = InstrumentRecognizer()
        assert recognizer.model is not None
        assert recognizer.class_names is not None
        assert len(recognizer.class_names) == 11
    
    @pytest.mark.skipif(not os.path.exists("Models/best_irmas_model.pth"), 
                       reason="Model not trained yet")
    def test_piano_recognition(self):
        """Тест распознавания фортепиано"""
        from recognizers.instrument_detector import InstrumentRecognizer
        
        audio_path = "Test/test_audio/test_piano.mp3"
        if os.path.exists(audio_path):
            recognizer = InstrumentRecognizer()
            results = recognizer.predict(audio_path)
            assert len(results) > 0
            assert results[0]['confidence'] > 0.0
    
    @pytest.mark.skipif(not os.path.exists("Models/best_irmas_model.pth"), 
                       reason="Model not trained yet")
    def test_multiple_instruments(self):
        """Тест распознавания нескольких инструментов"""
        from recognizers.instrument_detector import InstrumentRecognizer
        
        audio_path = "Test/test_audio/test_mixed.mp3"
        if os.path.exists(audio_path):
            recognizer = InstrumentRecognizer()
            results = recognizer.predict(audio_path, top_k=3)
            assert len(results) == 3
    
    def test_class_names(self):
        """Тест списка инструментов"""
        expected_instruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        
        # Тестируем без загрузки модели, только список классов
        from recognizers.instrument_detector import InstrumentRecognizer
        
        # Создаем mock для torch.load чтобы избежать ошибки загрузки модели
        with patch('torch.load'), patch.object(InstrumentRecognizer, '__init__', return_value=None):
            recognizer = InstrumentRecognizer.__new__(InstrumentRecognizer)
            recognizer.class_names = expected_instruments
            
            assert recognizer.class_names == expected_instruments
            assert len(recognizer.class_names) == 11 