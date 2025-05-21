
# 🧴 Bottle Detection с YOLOv5 и OpenCV

## Технологии:  
Python 3, PyTorch, YOLOv5, OpenCV

### Запуск проекта локально

1. Клонировать репозиторий YOLOv5 и перейти в папку проекта:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

2. Создать и активировать виртуальное окружение:
```bash
python3 -m venv venv
source venv/bin/activate    # Для Windows: venv\Scripts\activate
```

3. Установить зависимости:
```bash
pip install -r requirements.txt
pip install opencv-python
```

4. Проверить, что установлен PyTorch. Если нет, установить с помощью инструкции:  
https://pytorch.org/get-started/locally

5. Поместить `main.py` в корень проекта (рядом с папкой `yolov5`).

6. Запустить скрипт:
```bash
python main.py
```

---

Окно с видео с веб-камеры и обнаружением бутылок будет открыто автоматически.  
Для выхода нажмите `q`.
