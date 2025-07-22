import os
import os.path as osp
import sys
import time
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from ultralytics import YOLO
import yaml
from types import SimpleNamespace
from pathlib import Path

# --- Подключение трекеров ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'botsort'))
sys.path.append(os.path.join(ROOT, 'boxmot'))

from botsort.tracker.mc_bot_sort import BoTSORT as BoTSORT_ORIG
from boxmot.trackers.botsort.botsort import BotSort as BoTSORT_BOXMOT
from botsort.tracker.tracking_utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return SimpleNamespace(**config)

def write_results(filename, results):
    with open(filename, 'w') as f:
        for frame_id, tlbrs, ids, scores in results:
            for (x1, y1, x2, y2), tid, score in zip(tlbrs, ids, scores):
                w = x2 - x1
                h = y2 - y1
                f.write(f"{frame_id},{tid},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score:.6f},-1,-1,-1\n")
    logger.info(f"Results saved to {filename}")

def get_image_list(path):
    IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp"]
    files = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            if osp.splitext(name)[1].lower() in IMAGE_EXT:
                files.append(osp.join(root, name))
    return sorted(files)

def run_dual_tracking(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    model = YOLO(args.yolo_weights)
    model.fuse()
    
    tracker_orig = BoTSORT_ORIG(args, frame_rate=args.fps)
    if not args.with_reid:
        tracker_boxmot = BoTSORT_BOXMOT(args, device=device, half=args.fp16, frame_rate=args.fps, with_reid=args.with_reid)
    else:
        tracker_boxmot = BoTSORT_BOXMOT(
        reid_weights=Path(args.reid_weights),
        device=args.device_id,
        half=args.fp16,
        with_reid=args.with_reid,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
        cmc_method=args.cmc_method_boxmot,
        frame_rate=args.fps,
        fuse_first_associate=args.fuse_score,
        )



    timer = Timer()
    results_orig   = []
    results_boxmot = []

    # 3) Подготовка списка кадров
    if osp.isdir(args.input):
        image_paths = get_image_list(args.input)
        cap = None
    else:
        image_paths = None
        cap = cv2.VideoCapture(args.input)

    frame_id = 0
    while True: 
        frame_id += 1

        if frame_id == 10:
            break  # Для отладки, уберите это в реальном использовании
        
        # 3.1 Чтение кадра
        if image_paths:
            if frame_id > len(image_paths):
                break
            frame = cv2.imread(image_paths[frame_id - 1])
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # 4) Делаем детекции
        timer.tic()
        det_out = model(frame, conf=args.conf, imgsz=args.imgsz)[0]
        timer.toc()

        # 5) Формируем dets: (N,6) [x1,y1,x2,y2,score,class]
        if det_out.boxes is not None and len(det_out.boxes):
            xyxy  = det_out.boxes.xyxy.cpu().numpy()
            confs = det_out.boxes.conf.cpu().numpy()
            clss  = det_out.boxes.cls.cpu().numpy()
            dets  = np.concatenate([
                xyxy,
                confs[:, None],
                clss[:, None]
            ], axis=1)
        else:
            dets = np.zeros((0,6), dtype=float)

        # 6) Обновляем трекеры
        targets_orig   = tracker_orig.update(dets, frame)
        targets_boxmot = tracker_boxmot.update(dets, frame)

        # 7) Сбор результатов оригинального трекера (tlwh → tlbr)
        tlbrs_o, ids_o, scores_o = [], [], []
        for t in targets_orig:
            x, y, w, h = t.tlwh
            if w * h > 10:
                tlbrs_o.append((x, y, x + w, y + h))
                ids_o.append(t.track_id)
                scores_o.append(t.score)
        results_orig.append((frame_id, tlbrs_o, ids_o, scores_o))

        # 8) Сбор результатов BOXMOT (он уже отдаёт tlbr)
        tlbrs_b, ids_b, scores_b = [], [], []
        for t in targets_boxmot:
            x1, y1, x2, y2 = t[:4]
            tid = int(t[4])
            score = float(t[5])
            if (x2 - x1) * (y2 - y1) > 10:
                tlbrs_b.append((x1, y1, x2, y2))
                ids_b.append(tid)
                scores_b.append(score)
        results_boxmot.append((frame_id, tlbrs_b, ids_b, scores_b))


    # Запись результатов
    write_results(osp.join(args.save_dir, args.botsort_output), results_orig)
    write_results(osp.join(args.save_dir, args.boxmot_output), results_boxmot)

    if not image_paths:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_dual_tracking(args)
