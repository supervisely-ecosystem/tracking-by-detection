import os
import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import motmetrics as mm
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config

args = parse_args()

def load_tracks(fn, total_frames):
    """
    Загрузить предсказания из файла в формате:
    frame,id,x1,y1,w,h,conf,a,b,c
    и отфильтровать кадры > total_frames.
    """
    try:
        df = pd.read_csv(
            fn, header=None,
            names=['frame','id','x1','y1','w','h','conf','a','b','c']
        )
        dropped = df[df['frame'] > total_frames]
        if not dropped.empty:
            print(f"Предупреждение: в файле {fn} найдены фреймы > {total_frames} и будут проигнорированы.")
        df = df[df['frame'] <= total_frames]
        return df
    except Exception as e:
        raise ValueError(f"Ошибка при чтении {fn}: {e}")

def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(60, 255, 3).tolist())

def draw_boxes(image, gt_df, pred_df, frame_id, show_ids=False, show_gt=False):
    """
    Рисует на копии image:
     - GT-прямоугольники из gt_df (колонки x,y,w,h)
     - предсказания из pred_df (колонки x1,y1,w,h)
    """
    img = image.copy()

    # Рисуем GT
    if show_gt:
        for _, row in gt_df[gt_df.frame == frame_id].iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if show_ids:
                cv2.putText(
                    img, f'GT:{int(row.id)}',
                    (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

    # Рисуем предсказания
    for _, row in pred_df[pred_df.frame == frame_id].iterrows():
        x1, y1 = int(row.x1), int(row.y1)
        w,  h  = int(row.w),  int(row.h)
        x2, y2 = x1 + w,      y1 + h
        obj_id = int(row.id)
        color = get_color(obj_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if show_ids:
            cv2.putText(
                img, f'ID:{obj_id}',
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2
            )

    return img

def compute_metrics(gt_path, pred_path):
    gt   = mm.io.loadtxt(gt_path, fmt='mot15-2D', min_confidence=0)
    pred = mm.io.loadtxt(pred_path, fmt='mot15-2D')
    acc  = mm.MOTAccumulator(auto_id=True)

    frames = sorted(
        set(gt.index.get_level_values(0)) |
        set(pred.index.get_level_values(0))
    )

    for frame in frames:
        gt_frame   = gt.xs(frame, level=0, drop_level=False)   if frame in gt.index.get_level_values(0)   else pd.DataFrame()
        pred_frame = pred.xs(frame, level=0, drop_level=False) if frame in pred.index.get_level_values(0) else pd.DataFrame()

        gt_ids   = gt_frame.index.get_level_values(1).values   if not gt_frame.empty   else []
        pred_ids = pred_frame.index.get_level_values(1).values if not pred_frame.empty else []

        gt_boxes   = gt_frame[['X','Y','Width','Height']].values   if not gt_frame.empty   else []
        pred_boxes = pred_frame[['X','Y','Width','Height']].values if not pred_frame.empty else []

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames','idf1','idp','idr',
            'recall','precision','num_objects',
            'mostly_tracked','partially_tracked','mostly_lost',
            'num_false_positives','num_misses','num_switches',
            'mota','motp'
        ]
    )
    return summary

# ------------------------------------------------------------
# Загрузка кадров
frames = sorted([f for f in os.listdir(args['images']) if f.endswith('.jpg')])
if not frames:
    raise ValueError("В папке с изображениями нет .jpg файлов.")
sample_img = cv2.imread(os.path.join(args['images'], frames[0]))
h, w = sample_img.shape[:2]

# Загрузка GT
gt_df = pd.read_csv(
    args['gt'], header=None,
    names=['frame','id','x','y','w','h','conf','a','b','c']
)

# Загрузка предсказаний
predictions = {}
pred_dir = Path(args['predictions_dir'])
for file in os.listdir(pred_dir):
    if not file.endswith('.txt'):
        continue
    name = file[:-4]
    if not (name.isascii() and name.replace("_","").isalnum()):
        raise ValueError(f"Неверное имя файла предсказаний: {file}")
    path    = pred_dir / file
    pred_df = load_tracks(path, total_frames=len(frames))
    predictions[name] = pred_df

if not predictions:
    raise ValueError("Не найдено ни одного .txt файла предсказаний.")

# ------------------------------------------------------------
# Подготовка вывода видео
out_dir = Path(args['output_path'])
out_dir.mkdir(parents=True, exist_ok=True)
fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
video_writers = {}

# Индивидуальные видео
for name in predictions:
    video_writers[name] = cv2.VideoWriter(
        str(out_dir / f"{name}.mp4"), fourcc, 5, (w, h)
    )

# Коллаж, если нужно
make_collage = len(predictions) > 1
if make_collage:
    if len(predictions) == 2:
        shape = 'vertical' if w > h else 'horizontal'
        size  = (w, h*2) if shape=='vertical' else (w*2, h)
    else:
        size = (w*2, h*2)
    video_writers['collage'] = cv2.VideoWriter(
        str(out_dir / "comparison.mp4"), fourcc, 5, size
    )

# ------------------------------------------------------------
# Визуализация по кадрам
for idx, fname in enumerate(frames, start=1):
    img = cv2.imread(os.path.join(args['images'], fname))
    panels = {}

    for name, df in predictions.items():
        panel = draw_boxes(
            img, gt_df, df, idx,
            show_ids=args['show-model-names'],
            show_gt =args['show-gt']
        )
        if args['show-model-names']:
            cv2.putText(
                panel, name, (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0,0,0), 5, cv2.LINE_AA
            )
            cv2.putText(
                panel, name, (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255,255,255),2, cv2.LINE_AA
            )
        panels[name] = panel
        video_writers[name].write(panel)

    if 'collage' in video_writers:
        imgs = list(panels.values())
        if len(imgs) == 2:
            collage = np.vstack(imgs) if shape=='vertical' else np.hstack(imgs)
        else:
            while len(imgs) < 4:
                imgs.append(np.zeros_like(img))
            top    = np.hstack((imgs[0], imgs[1]))
            bottom = np.hstack((imgs[2], imgs[3]))
            collage = np.vstack((top, bottom))
        video_writers['collage'].write(collage)

# Закрываем видеофайлы
for writer in video_writers.values():
    writer.release()

print(f"Видео сохранены в: {out_dir.resolve()}")

# ------------------------------------------------------------
# Расчёт метрик
print("Вычисление метрик...")
results = []
for name in predictions:
    pred_path = pred_dir / f"{name}.txt"
    summary   = compute_metrics(args['gt'], pred_path)
    summary['tracker'] = name
    results.append(summary)

combined = pd.concat(results).reset_index(drop=True)
combined = combined[[
    'tracker','mota','idf1','precision','recall',
    'num_switches','num_false_positives','num_misses'
]]

excel_path = out_dir / "metrics.xlsx"
combined.to_excel(excel_path, index=False)
print(f"Метрики сохранены в: {excel_path}")
