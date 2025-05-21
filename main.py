import cv2
import torch

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Запуск камеры (0 — встроенная камера)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получение предсказаний
    results = model(frame)

    # Извлекаем таблицу с результатами
    detections = results.pandas().xyxy[0]

    # Фильтрация по классу 'bottle'
    bottles = detections[detections['name'] == 'bottle']

    # Рисуем рамки вокруг бутылок
    for *box, conf, cls, name in bottles.values:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{name} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Показываем кадр
    cv2.imshow('Bottle Detection (Webcam)', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
