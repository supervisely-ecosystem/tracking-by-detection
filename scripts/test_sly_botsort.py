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

# Импорт обёртки Supervisely - используем НОВЫЙ трекер
from supervisely import Annotation, Label, Rectangle, ObjClass, ProjectMeta
from supervisely.nn.tracker.botsort.sly_tracker import BoTTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return SimpleNamespace(**config)

def write_results(filename, results):
    """Записывает результаты в MOT формате"""
    with open(filename, 'w') as f:
        for frame_id, tlbrs, ids, scores in results:
            for (x1, y1, x2, y2), tid, score in zip(tlbrs, ids, scores):
                w = x2 - x1
                h = y2 - y1
                f.write(f"{frame_id},{tid},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score:.6f},-1,-1,-1\n")
    logger.info(f"Results saved to {filename}")

def get_image_list(path):
    """Получение списка изображений из папки"""
    IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp"]
    files = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            if osp.splitext(name)[1].lower() in IMAGE_EXT:
                files.append(osp.join(root, name))
    return sorted(files)

def create_obj_classes(class_names):
    """Создание классов объектов для Supervisely"""
    obj_classes = {}
    for class_name in class_names:
        obj_classes[class_name] = ObjClass(class_name, Rectangle)
    return obj_classes

def detections_to_annotation(dets, obj_classes, img_size, conf_threshold=0.3):
    """
    Конвертация детекций YOLO в Supervisely Annotation
    
    Args:
        dets: numpy array (N,6) [x1,y1,x2,y2,score,class]
        obj_classes: dict с классами объектов
        img_size: (height, width)
        conf_threshold: минимальный порог уверенности
    
    Returns:
        Annotation: объект аннотации Supervisely
    """
    labels = []
    
    for det in dets:
        if len(det) < 6:
            continue
            
        x1, y1, x2, y2, score, cls_id = det
        
        # Фильтрация по уверенности
        if score < conf_threshold:
            continue
            
        # Получение имени класса (предполагаем COCO классы)
        cls_id = int(cls_id)
        if cls_id in COCO_CLASSES:
            class_name = COCO_CLASSES[cls_id]
        else:
            class_name = f"class_{cls_id}"
            
        # Создание класса если его нет
        if class_name not in obj_classes:
            obj_classes[class_name] = ObjClass(class_name, Rectangle)
            
        # Создание геометрии
        geometry = Rectangle(
            top=int(y1), 
            left=int(x1), 
            bottom=int(y2), 
            right=int(x2)
        )
        
        # Создание метки
        label = Label(geometry=geometry, obj_class=obj_classes[class_name])
        labels.append(label)
    
    return Annotation(img_size=img_size, labels=labels)

def extract_tracks_from_video_annotation(video_ann):
    """
    Извлечение треков из VideoAnnotation в MOT формат
    
    Returns:
        List[Tuple]: (frame_id, tlbrs, ids, scores)
    """
    results = []
    uuid_to_int = {}
    next_int_id = 1
    
    for frame_id, frame in enumerate(video_ann.frames, 1):
        tlbrs, ids, scores = [], [], []
        
        # Frame содержит figures, а не labels
        for figure in frame.figures:
            # Получение координат из геометрии фигуры
            geom = figure.geometry
            x1, y1 = geom.left, geom.top
            x2, y2 = geom.right, geom.bottom
            
            # Получение track_id из video_object
            # track_id = figure.video_object.key() if figure.video_object else None
            try:
                uuid_key = figure.video_object.key() 
                if uuid_key not in uuid_to_int:
                    uuid_to_int[uuid_key] = next_int_id
                    next_int_id += 1
            
                track_id = uuid_to_int[uuid_key]
                
            except Exception as e:
                print(f"Error converting track_id: {e}")
                track_id = None     

            score = 1.0  # По умолчанию
            
            # Если есть теги у фигуры, ищем confidence
            if hasattr(figure, 'tags') and figure.tags:
                for tag in figure.tags:
                    if tag.name in ['confidence', 'score', 'conf']:
                        score = float(tag.value)
                        break
            
            if track_id is not None:
                tlbrs.append((x1, y1, x2, y2))
                ids.append(track_id)
                scores.append(score)
        
        results.append((frame_id, tlbrs, ids, scores))
    
    return results

def run_supervisely_tracking(args):
    """Основная функция трекинга с использованием Supervisely BoTTracker"""
    
    # Отладочная информация
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    device = getattr(args, 'device', "cuda" if torch.cuda.is_available() else "cpu")
    
    # Проверяем доступность CUDA устройства
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and torch.cuda.is_available():
        device_id = getattr(args, 'device_id', '0')
        if isinstance(device_id, str) and device_id.isdigit():
            device_idx = int(device_id)
            if device_idx >= torch.cuda.device_count():
                logger.warning(f"GPU {device_idx} not available, using GPU 0")
                device = "cuda:0"
            else:
                device = f"cuda:{device_idx}"
        else:
            device = "cuda:0"
    
    logger.info(f"Using device: {device}")
    
    # Создание директории для результатов
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Загрузка YOLO модели
    model = YOLO(args.yolo_weights)
    model.fuse()
    
    # Создание классов объектов
    obj_classes = create_obj_classes(list(COCO_CLASSES.values()))
    
    # Настройки для BoTTracker из конфига
    tracker_settings = {
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "new_track_thresh": args.new_track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "min_box_area": getattr(args, 'min_box_area', 10.0),
        "fuse_score": getattr(args, 'fuse_score', False),
        "with_reid": args.with_reid,
        "proximity_thresh": args.proximity_thresh,
        "appearance_thresh": args.appearance_thresh,
        "cmc_method": args.cmc_method,
        "device": device,  # Передаем устройство явно
        "device_id": getattr(args, 'device_id', '0'),  # Добавляем device_id
    }
    
    # Добавляем ReID параметры если включен
    if args.with_reid:
        tracker_settings.update({
            "reid_model": getattr(args, 'reid_model', 'osnet_reid'),
            "reid_weights": getattr(args, 'reid_weights', None),
            "fast_reid_config": getattr(args, 'fast_reid_config', None),
            "fast_reid_weights": getattr(args, 'fast_reid_weights', None),
        })
    
    # Инициализация трекера с обработкой ошибок
    try:
        tracker = BoTTracker(settings=tracker_settings)
        logger.info("Supervisely BoTTracker initialized successfully")
        logger.info(f"Tracker settings: {tracker_settings}")
    except Exception as e:
        logger.error(f"Failed to initialize BoTTracker: {e}")
        logger.info("Trying with minimal settings...")
        
        # Попробуем с минимальными настройками
        minimal_settings = {
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.7,
            "with_reid": False,  # Отключаем ReID для отладки
            "device": "cpu" if "cuda" in str(e).lower() else device,
        }
        
        try:
            tracker = BoTTracker(settings=minimal_settings)
            logger.info("BoTTracker initialized with minimal settings")
        except Exception as e2:
            logger.error(f"Failed to initialize BoTTracker even with minimal settings: {e2}")
            raise
    
    # Подготовка входных данных
    if osp.isdir(args.input):
        image_paths = get_image_list(args.input)
        frames_source = image_paths
        logger.info(f"Found {len(image_paths)} images in directory: {args.input}")
    else:
        frames_source = args.input
        logger.info(f"Using video file: {args.input}")
    
    # Получение кадров и детекций
    frame_to_annotation = {}
    frames_list = []
    
    # Чтение кадров
    if isinstance(frames_source, list):  # Список изображений
        for i, img_path in enumerate(frames_source):
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            frames_list.append(frame)
            
            # Получение детекций
            det_out = model(frame, conf=args.conf, imgsz=args.imgsz)[0]
            
            # Формирование детекций
            if det_out.boxes is not None and len(det_out.boxes):
                xyxy = det_out.boxes.xyxy.cpu().numpy()
                confs = det_out.boxes.conf.cpu().numpy()
                clss = det_out.boxes.cls.cpu().numpy()
                dets = np.concatenate([xyxy, confs[:, None], clss[:, None]], axis=1)
            else:
                dets = np.zeros((0, 6), dtype=float)
            
            # Конвертация в аннотацию
            img_size = (frame.shape[0], frame.shape[1])
            annotation = detections_to_annotation(dets, obj_classes, img_size, args.conf)
            frame_to_annotation[i] = annotation
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed frame {i+1}/{len(frames_source)}, detections: {len(annotation.labels)}")
    
    else:  # Видео файл
        cap = cv2.VideoCapture(frames_source)
        if not cap.isOpened():
            logger.error(f"Could not open video: {frames_source}")
            return
            
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_list.append(frame)
            
            # Получение детекций
            det_out = model(frame, conf=args.conf, imgsz=args.imgsz)[0]
            
            # Формирование детекций
            if det_out.boxes is not None and len(det_out.boxes):
                xyxy = det_out.boxes.xyxy.cpu().numpy()
                confs = det_out.boxes.conf.cpu().numpy()
                clss = det_out.boxes.cls.cpu().numpy()
                dets = np.concatenate([xyxy, confs[:, None], clss[:, None]], axis=1)
            else:
                dets = np.zeros((0, 6), dtype=float)
            
            # Конвертация в аннотацию
            img_size = (frame.shape[0], frame.shape[1])
            annotation = detections_to_annotation(dets, obj_classes, img_size, args.conf)
            frame_to_annotation[frame_id] = annotation
            
            frame_id += 1
            if frame_id % 50 == 0:
                logger.info(f"Processed frame {frame_id}, detections: {len(annotation.labels)}")
        
        cap.release()
        logger.info(f"Total frames read from video: {frame_id}")
    
    if not frames_list:
        logger.error("No frames were processed!")
        return
    
    # Определение размера кадра
    frame_shape = (frames_list[0].shape[0], frames_list[0].shape[1])
    
    logger.info(f"Frame shape: {frame_shape}")
    logger.info(f"Total frames: {len(frame_to_annotation)}")
    
    # Запуск трекинга
    logger.info("Starting Supervisely BoTSORT tracking...")
    start_time = time.time()
    
    try:
        video_annotation = tracker.track(
            source=frames_list,
            frame_to_annotation=frame_to_annotation,
            frame_shape=frame_shape
        )
        
        tracking_time = time.time() - start_time
        logger.info(f"Tracking completed in {tracking_time:.2f} seconds")
        
        # Извлечение результатов
        results = extract_tracks_from_video_annotation(video_annotation)
        
        # Сохранение результатов - используем имя из конфига или дефолтное
        output_filename = getattr(args, 'botsort_output', 'supervisely_botsort.txt')
        output_path = osp.join(args.save_dir, output_filename)
        write_results(output_path, results)
        
        # Статистика
        total_tracks = set()
        total_detections = 0
        for _, _, ids, _ in results:
            total_tracks.update(ids)
            total_detections += len(ids)
        
        logger.info(f"Tracking statistics:")
        logger.info(f"  - Total frames processed: {len(results)}")
        logger.info(f"  - Total unique tracks: {len(total_tracks)}")
        logger.info(f"  - Total detections: {total_detections}")
        logger.info(f"  - Average detections per frame: {total_detections/len(results) if len(results) > 0 else 0:.2f}")
        logger.info(f"Tracking completed successfully. Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        import traceback
        traceback.print_exc()
        raise

# Словарь COCO классов
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

if __name__ == "__main__":
    args = parse_args()
    run_supervisely_tracking(args)