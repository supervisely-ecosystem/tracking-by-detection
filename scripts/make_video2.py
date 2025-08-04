import cv2
import os
from natsort import natsorted

# Путь к папке с изображениями
image_folder = 'data/dataset/train/v_-6Os86HzwCs_c009/img1/'  # например: './frames'
output_video = 'data/dataset/train/v_-6Os86HzwCs_c009/output.avi'
fps = 30  # Частота кадров

# Получаем список изображений и сортируем их
images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))]
images = natsorted(images)  # Натуральная сортировка (например, 1.jpg, 2.jpg, 10.jpg)

if not images:
    raise ValueError("Нет изображений в указанной папке.")

# Читаем первое изображение, чтобы получить размер
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, _ = frame.shape

# Создаём объект VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Склеиваем изображения
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)
    resized_frame = cv2.resize(frame, (width, height))
    video.write(resized_frame)

video.release()
print(f"Видео сохранено как: {output_video}")
