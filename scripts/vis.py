import os
import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config

# parser = argparse.ArgumentParser()
# parser.add_argument('--images', required=True)
# parser.add_argument('--gt', required=True)
# parser.add_argument('--predictions_dir', required=True)
# parser.add_argument('--output_path', default='data/vis_output')
# parser.add_argument('--show-model-names', action='store_true')
# parser.add_argument('--show-ids', action='store_true')
# parser.add_argument('--show-gt', action='store_true')
# parser.add_argument('--save-mp4', action='store_true', default=True)
# args = parser.parse_args()
args = parse_args()

def load_tracks(fn, total_frames):
    try:
        df = pd.read_csv(fn, header=None, names=['frame','id','x1','y1','x2','y2','conf','a','b','c'])
        dropped = df[df['frame'] > total_frames]
        if not dropped.empty:
            print(f"Предупреждение: в файле {fn} найдены фреймы, выходящие за пределы изображений и будут проигнорированы.")
        df = df[df['frame'] <= total_frames]
        return df
    except Exception as e:
        raise ValueError(f"Ошибка при чтении {fn}: {e}")

def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(60, 255, 3).tolist())

def draw_boxes(image, gt_df, pred_df, frame_id, show_ids=False, show_gt=False):
    img = image.copy()
    if show_gt:
        for _, row in gt_df[gt_df.frame == frame_id].iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if show_ids:
                cv2.putText(img, f'GT:{int(row.id)}', (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for _, row in pred_df[pred_df.frame == frame_id].iterrows():
        x1, y1, x2, y2, obj_id = int(row.x1), int(row.y1), int(row.x2), int(row.y2), int(row.id)
        color = get_color(obj_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if show_ids:
            cv2.putText(img, f'ID:{obj_id}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Загрузка изображений
frames = sorted([f for f in os.listdir(args.images) if f.endswith('.jpg')])
if not frames:
    raise ValueError("В указанной папке с изображениями не найдено .jpg файлов.")

sample_img = cv2.imread(os.path.join(args.images, frames[0]))
h, w = sample_img.shape[:2]
out_h, out_w = h * 2, w * 2

# Загрузка GT
gt_df = pd.read_csv(args.gt, header=None, names=['frame','id','x','y','w','h','conf','a','b','c'])

# Загрузка предсказаний
predictions = {}
pred_dir = Path(args.predictions_dir)
for file in os.listdir(pred_dir):
    if not file.endswith('.txt'):
        continue
    name = file[:-4]
    if not name.isascii() or not name.replace("_", "").isalnum():
        raise ValueError(f"Недопустимое имя файла предсказаний: {file}. Используйте только латиницу, цифры и подчёркивания.")
    path = pred_dir / file
    pred_df = load_tracks(path, total_frames=len(frames))
    predictions[name] = pred_df

if not predictions:
    raise ValueError("Не найдено ни одного .txt файла предсказаний в указанной директории.")

# Выходная директория
out_dir = Path(args.output_path)
out_dir.mkdir(parents=True, exist_ok=True)

# Видео кодек
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writers = {}

# Индивидуальные видео
for name in predictions:
    video_writers[name] = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), fourcc, 5, (w, h))

# Коллаж, если моделей > 1
make_collage = len(predictions) > 1
if make_collage:
    collage_shape = None
    if len(predictions) == 2:
        collage_shape = 'vertical' if w > h else 'horizontal'
        if collage_shape == 'vertical':
            collage_size = (w, h * 2)
        else:
            collage_size = (w * 2, h)
    else:
        collage_size = (w * 2, h * 2)
    video_writers['collage'] = cv2.VideoWriter(str(out_dir / "comparison.mp4"), fourcc, 5, collage_size)

# Основной цикл
for idx, fname in enumerate(frames, start=1):
    img = cv2.imread(os.path.join(args.images, fname))
    panels = {}

    for name, df in predictions.items():
        panel = draw_boxes(img, gt_df, df, idx, show_ids=args.show_ids, show_gt=args.show_gt)
        if args.show_model_names:
            # Чёрная обводка
            cv2.putText(panel, name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            # Белый текст
            cv2.putText(panel, name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        panels[name] = panel
        video_writers[name].write(panel)

    if 'collage' in video_writers:
        images = list(panels.values())
        if len(images) == 2:
            if collage_shape == 'vertical':
                collage = np.vstack((images[0], images[1]))
            else:
                collage = np.hstack((images[0], images[1]))
        else:
            while len(images) < 4:
                images.append(np.zeros_like(img))
            top = np.hstack((images[0], images[1]))
            bottom = np.hstack((images[2], images[3]))
            collage = np.vstack((top, bottom))
        video_writers['collage'].write(collage)

# Завершение записи
for writer in video_writers.values():
    writer.release()

print(f"Видео успешно сохранены в папку: {out_dir.resolve()}")
