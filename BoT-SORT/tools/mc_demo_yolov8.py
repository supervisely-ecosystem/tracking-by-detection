# tools/mc_demo_yolov8.py

import sys
import argparse
import os
import os.path as osp
import time
from pathlib import Path
import gc

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from loguru import logger

from yolox.utils.visualize import plot_tracking
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT + YOLOv8 Demo")
    parser.add_argument("demo", choices=["image", "images", "video", "webcam"], help="demo type")
    parser.add_argument("-n", "--name", type=str, default="yolov8_demo", help="model name")
    parser.add_argument("-c", "--ckpt", type=str, default="yolov8n.pt", help="path to YOLOv8 weights")
    parser.add_argument("--path", type=str, default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam camera id")
    parser.add_argument("--save_result", action="store_true", help="save result images/videos")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], help="device to run on")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--fps", type=int, default=30, help="frame rate for tracking")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6)
    parser.add_argument("--track_low_thresh", type=float, default=0.1)
    parser.add_argument("--new_track_thresh", type=float, default=0.7)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--min_box_area", type=float, default=10)
    parser.add_argument("--fuse-score", dest="fuse_score", action="store_true")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", 
                        help="cmc method: sparseOptFlow | files | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", action="store_true")
    parser.add_argument("--fast-reid-config", default="fast_reid/configs/MOT17/sbs_S50.yml")
    parser.add_argument("--fast-reid-weights", default="pretrained/mot17_sbs_S50.pth")
    parser.add_argument("--proximity_thresh", type=float, default=0.5)
    parser.add_argument("--appearance_thresh", type=float, default=0.25)
    parser.add_argument("--fp16", action="store_true", help="use mixed precision (half)")

    return parser


def get_image_list(path):
    images = []
    for root, _, files in os.walk(path):
        for f in files:
            if osp.splitext(f)[1].lower() in IMAGE_EXT:
                images.append(osp.join(root, f))
    return sorted(images)


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


class Predictor:
    def __init__(self, ckpt, device, conf, imgsz, fp16=False):
        self.model = YOLO(ckpt)
        if device == "gpu" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.model.to("cpu")
        self.model.fuse()
        self.conf = conf
        self.imgsz = imgsz
        self.fp16 = fp16

    def inference(self, img, timer):
        # img: path or ndarray
        if isinstance(img, str):
            frame = cv2.imread(img)
        else:
            frame = img.copy()
        img_info = {"raw_img": frame}

        timer.tic()
        # YOLOv8 returns Boxes in orig image scale
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, half=self.fp16)[0]
        timer.toc()

        dets = []
        boxes = results.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
            confs = boxes.conf.cpu().numpy()  # (N,)
            clss  = boxes.cls.cpu().numpy()   # (N,)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                c = confs[i]
                cls = clss[i]
                # replicate obj_conf and cls_conf for compatibility
                dets.append([x1, y1, x2, y2, c, c, cls])

        return np.array(dets), img_info


def image_demo(predictor, vis_folder, current_time, args):
    files = get_image_list(args.path) if osp.isdir(args.path) else [args.path]
    tracker = BoTSORT(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, p in enumerate(files, 1):
        gc.collect()    
        dets, img_info = predictor.inference(p, timer)
        online_targets = tracker.update(dets, img_info["raw_img"])

        tlwhs, ids, scores = [], [], []
        for t in online_targets:
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] > args.min_box_area:
                tlwhs.append(tlwh)
                ids.append(t.track_id)
                scores.append(t.score)
        results.append((frame_id, tlwhs, ids, scores))

        im = plot_tracking(img_info["raw_img"], tlwhs, ids,
                           frame_id=frame_id,
                           fps=1. / timer.average_time)

        if args.save_result:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_dir = osp.join(vis_folder, ts)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(osp.join(save_dir, osp.basename(p)), im)

        if cv2.waitKey(1) in [27, ord('q')]:
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{ts}.txt")
        write_results(res_file, results)


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo=="video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, ts)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1] if args.demo=="video" else "cam.mp4")
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    tracker = BoTSORT(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        dets, img_info = predictor.inference(frame, timer)
        online_targets = tracker.update(dets, frame)
        tlwhs, ids, scores = [], [], []
        for t in online_targets:
            tlwh = t.tlwh
            if tlwh[2]*tlwh[3] > args.min_box_area:
                tlwhs.append(tlwh)
                ids.append(t.track_id)
                scores.append(t.score)
        results.append((frame_id, tlwhs, ids, scores))

        im = plot_tracking(frame, tlwhs, ids, frame_id=frame_id, fps=1./timer.average_time)
        writer.write(im)
        cv2.imshow("Tracking", im)
        if cv2.waitKey(1) in [27, ord('q')]:
            break

    cap.release()
    writer.release()
    if args.save_result:
        res_file = osp.join(save_folder, f"{ts}.txt")
        write_results(res_file, results)


def main(args):
    # mimic original
    args.ablation = False
    args.mot20 = not args.fuse_score
    args.name = args.name or "yolov8_demo"

    device = "cuda" if args.device=="gpu" and torch.cuda.is_available() else "cpu"
    predictor = Predictor(args.ckpt, device, args.conf, args.imgsz, args.fp16)

    if args.save_result:
        vis_folder = osp.join("runs", args.name)
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = None

    current_time = time.localtime()
    if args.demo in ["image", "images"]:
        image_demo(predictor, vis_folder, current_time, args)
    else:
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.ablation = False
    args.mot20 = not args.fuse_score
    # ReID несовместим с fp16
    if args.with_reid:
        args.fp16 = False

    main(args)
