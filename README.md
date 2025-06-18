
## Быстрый старт

1. **Клонируйте репозиторий:**
   ```sh
   git clone https://github.com/BoDy-PrISon/musicnow-api.git
   cd musicnow-api
   ```

2. **Создайте и активируйте виртуальное окружение:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac
   ```

3. **Установите зависимости:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Запустите сервер:**
   ```sh
   python main.py
   ```

## Использование API

- **POST /recognize** — отправьте аудиофайл для распознавания жанра, инструментов и настроения.
- **GET /tracks** — получить список сохранённых треков.
- **Swagger/OpenAPI** — документация доступна по адресу `/docs` (если используется FastAPI/Swagger).

## Обучение моделей

Скрипты для обучения находятся в папке `Train/`:
- `Multiclass_Train.py` — обучение жанровой классификации
- `MultiLabel_Train.py` — обучение инструментальной классификации
- `Mood_Train.py` — обучение модели настроения

## Визуализация архитектуры

Для презентации архитектуры моделей используйте скрипт `Visualaze.py`.  
Генерируются схемы в PNG с подписями на русском языке.
