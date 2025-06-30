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

# --- Подключение трекеров ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'botsort'))
sys.path.append(os.path.join(ROOT, 'boxmot', 'boxmot'))

from botsort.tracker.mc_bot_sort import BoTSORT as BoTSORT_ORIG
from boxmot.trackers.botsort.botsort import BotSort as BoTSORT_BOXMOT
from botsort.yolox.utils.visualize import plot_tracking
from botsort.tracker.tracking_utils.timer import Timer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Путь до видео или папки с изображениями")
    parser.add_argument("-n", "--name", type=str, default="yolov8_demo", help="model name")
    parser.add_argument("--yolo_weights", type=str, default="yolov8n.pt", help="YOLOv8 веса")
    parser.add_argument("--imgsz", type=int, default=640, help="размер изображения")
    parser.add_argument("--conf", type=float, default=0.3, help="порог уверенности YOLO")
    parser.add_argument("--save_dir", type=str, default="runs", help="Папка для сохранения результатов")
    parser.add_argument("--fps", type=int, default=30)

    # tracking params
    parser.add_argument("--track_high_thresh", type=float, default=0.6)
    parser.add_argument("--track_low_thresh", type=float, default=0.1)
    parser.add_argument("--new_track_thresh", type=float, default=0.7)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--min_box_area", type=float, default=10)

    # appearance model (reid)
    parser.add_argument("--with-reid", action="store_true")
    parser.add_argument("--fast-reid-config", default="", help="путь до .yml конфигурации fastreid")
    parser.add_argument("--fast-reid-weights", default="", help="путь до весов fastreid")
    parser.add_argument("--proximity_thresh", type=float, default=0.5)
    parser.add_argument("--appearance_thresh", type=float, default=0.25)

    parser.add_argument("--fuse_score", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cmc_method", default="sparseOptFlow", help="Метод компенсации движения камеры: sparseOptFlow | ecc | orb | files")


    return parser.parse_args()


def write_results(filename, results):
    fmt = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, tid, sc in zip(tlwhs, track_ids, scores):
                x, y, w, h = tlwh
                f.write(fmt.format(frame=frame_id, id=tid,
                                   x1=round(x, 1), y1=round(y, 1),
                                   w=round(w, 1), h=round(h, 1),
                                   s=round(sc, 2)))
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
    tracker_boxmot = BoTSORT_BOXMOT(args, device=device, half=args.fp16, frame_rate=args.fps, with_reid=args.with_reid)

    timer = Timer()
    results_orig = []
    results_boxmot = []

    # Определим режим: видео или изображения
    if osp.isdir(args.input):
        image_paths = get_image_list(args.input)
    else:
        cap = cv2.VideoCapture(args.input)
        image_paths = None

    frame_id = 0
    while True:
        frame_id += 1
        if image_paths:
            if frame_id > len(image_paths):
                break
            frame = cv2.imread(image_paths[frame_id - 1])
        else:
            ret, frame = cap.read()
            if not ret:
                break

        timer.tic()
        results = model(frame, conf=args.conf, imgsz=args.imgsz)[0]
        timer.toc()

        dets = []
        if results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss  = results.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                c = confs[i]
                cls = clss[i]
                dets.append([x1, y1, x2, y2, c, c, cls])
        dets = np.array(dets)

        # Обновляем оба трекера
        targets_orig = tracker_orig.update(dets, frame)
        targets_boxmot = tracker_boxmot.update(dets, frame)

        # for targets, results_list in zip([targets_orig, targets_boxmot], [results_orig, results_boxmot]):
        #     tlwhs, ids, scores = [], [], []
        #     for t in targets:
        #         tlwh = t.tlwh
        #         if tlwh[2] * tlwh[3] > 10:  # min_box_area
        #             tlwhs.append(tlwh)
        #             ids.append(t.track_id)
        #             scores.append(t.score)
        #     results_list.append((frame_id, tlwhs, ids, scores))
        
                # трекер botsort (оригинал)
        tlwhs_o, ids_o, scores_o = [], [], []
        for t in targets_orig:
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] > 10:
                tlwhs_o.append(tlwh)
                ids_o.append(t.track_id)
                scores_o.append(t.score)
        results_orig.append((frame_id, tlwhs_o, ids_o, scores_o))

        # трекер boxmot
        tlwhs_b, ids_b, scores_b = [], [], []
        for t in targets_boxmot:
            tlwh = t[:4]
            tid = int(t[4])
            score = float(t[5])
            if tlwh[2] * tlwh[3] > 10:
                tlwhs_b.append(tlwh)
                ids_b.append(tid)
                scores_b.append(score)
        results_boxmot.append((frame_id, tlwhs_b, ids_b, scores_b))


        # Отрисовка
        vis = plot_tracking(frame.copy(), [t.tlwh for t in targets_orig], [t.track_id for t in targets_orig],
                            frame_id=frame_id, fps=1. / max(0.001, timer.average_time))
        cv2.imshow("BoT-SORT original", vis)
        if cv2.waitKey(1) == 27:
            break

    # Запись результатов
    write_results(osp.join(args.save_dir, "botsort_original.txt"), results_orig)
    write_results(osp.join(args.save_dir, "botsort_boxmot.txt"), results_boxmot)

    if not image_paths:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_dual_tracking(args)
