#!/usr/bin/env python3
"""
Скрипт для сборки набора кадров в видеофайл с помощью OpenCV.
"""
import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Convert image frames to video"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to directory with image frames"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output video file, e.g. output.mp4"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second"
    )
    parser.add_argument(
        "--ext", default="jpg",
        help="Image file extension (jpg, png, etc.)"
    )
    args = parser.parse_args()

    # Собираем список файлов и сортируем
    images = [f for f in os.listdir(args.input_dir) if f.endswith(args.ext)]
    images.sort()

    if not images:
        print(f"No images found with extension '{args.ext}' in {args.input_dir}")
        return

    # Читаем первый кадр для получения размера
    first_path = os.path.join(args.input_dir, images[0])
    first_frame = cv2.imread(first_path)
    if first_frame is None:
        print(f"Failed to read the first frame: {first_path}")
        return
    height, width = first_frame.shape[:2]

    # Определяем кодек и создаём VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # для .mp4
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    # Проходим по всем кадрам и записываем
    for img_name in images:
        img_path = os.path.join(args.input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: could not read '{img_path}'")
            continue
        writer.write(frame)

    writer.release()
    print(f"Video saved to {args.output}")

if __name__ == "__main__":
    main()