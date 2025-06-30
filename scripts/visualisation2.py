import cv2, os
import pandas as pd
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True)
parser.add_argument('--gt', required=True)
parser.add_argument('--botsort', required=True)
parser.add_argument('--boxmot', required=True)
parser.add_argument('--botsort-reid', required=True)
parser.add_argument('--boxmot-reid', required=True)
parser.add_argument('--output_path', default='data/vis_output')
parser.add_argument('--show-model-names', action='store_true')
parser.add_argument('--show-ids', action='store_true')
parser.add_argument('--save-mp4', action='store_true', default=True)

args = parser.parse_args()

def load_tracks(fn):
    return pd.read_csv(fn, header=None, names=['frame','id','x','y','w','h','conf','a','b','c'])

def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(60, 255, 3).tolist())

def draw_boxes(image, gt_df, pred_df, frame_id, show_ids=False):
    img = image.copy()
    for _, row in gt_df[gt_df.frame == frame_id].iterrows():
        x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if show_ids:
            cv2.putText(img, f'GT:{int(row.id)}', (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for _, row in pred_df[pred_df.frame == frame_id].iterrows():
        x, y, w, h, obj_id = int(row.x), int(row.y), int(row.w), int(row.h), int(row.id)
        color = get_color(obj_id)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        if show_ids:
            cv2.putText(img, f'ID:{obj_id}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Загрузка данных
gt = load_tracks(args.gt)
botsort = load_tracks(args.botsort)
boxmot = load_tracks(args.boxmot)
botsort_reid = load_tracks(args.botsort_reid)
boxmot_reid = load_tracks(args.boxmot_reid)

frames = sorted([f for f in os.listdir(args.images) if f.endswith('.jpg')])
sample_img = cv2.imread(os.path.join(args.images, frames[0]))
h, w = sample_img.shape[:2]
out_h, out_w = h * 2, w * 2

# Выходная папка
out_dir = Path(args.output_path)
out_dir.mkdir(exist_ok=True)

# Видео кодек
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Видео для каждой модели
video_writers = {
    'botsort':        cv2.VideoWriter(str(out_dir/'botsort.mp4'), fourcc, 5, (w, h)),
    'boxmot':         cv2.VideoWriter(str(out_dir/'boxmot.mp4'), fourcc, 5, (w, h)),
    'botsort_reid':   cv2.VideoWriter(str(out_dir/'botsort_reid.mp4'), fourcc, 5, (w, h)),
    'boxmot_reid':    cv2.VideoWriter(str(out_dir/'boxmot_reid.mp4'), fourcc, 5, (w, h)),
    'collage':        cv2.VideoWriter(str(out_dir/'comparison.mp4'), fourcc, 5, (out_w, out_h)),
}

for idx, fname in enumerate(frames, start=1):
    img = cv2.imread(os.path.join(args.images, fname))

    panels = {
        'botsort':      draw_boxes(img, gt, botsort, idx, args.show_ids),
        'botsort_reid': draw_boxes(img, gt, botsort_reid, idx, args.show_ids),
        'boxmot':       draw_boxes(img, gt, boxmot, idx, args.show_ids),
        'boxmot_reid':  draw_boxes(img, gt, boxmot_reid, idx, args.show_ids),
    }

    # Подписи — после рисования, чтобы не размылись при resize
    if args.show_model_names:
        cv2.putText(panels['botsort'],      'BotSORT',        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        cv2.putText(panels['boxmot'],       'BoxMOT',         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        cv2.putText(panels['botsort_reid'], 'BotSORT+ReID',   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        cv2.putText(panels['boxmot_reid'],  'BoxMOT+ReID',    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    # Запись отдельных видео
    for k, writer in video_writers.items():
        if k in panels:
            writer.write(panels[k])

    # Коллаж
    top = np.hstack((panels['botsort'], panels['boxmot']))
    bottom = np.hstack((panels['botsort_reid'], panels['boxmot_reid']))
    collage = np.vstack((top, bottom))

    video_writers['collage'].write(collage)

# Завершаем запись
for writer in video_writers.values():
    writer.release()

print(f"Видео сохранены в: {out_dir.resolve()}")
