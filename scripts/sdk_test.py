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
import yaml
from types import SimpleNamespace


# Supervisely imports
import supervisely as sly
from supervisely import Annotation, Label, Rectangle, ObjClass, ProjectMeta
from supervisely.nn.inference import Session
from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT as BoTSORT_ORIG
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


def create_supervisely_detector(args):
    """Create detector Supervisely API"""
    # Инициализация API (токен должен быть в переменных окружения)
    api = sly.Api()
    
    if hasattr(args, 'detector_task_id') and args.detector_task_id:
        # Использование уже развернутой модели по task_id
        logger.info(f"Using deployed detector with task_id: {args.detector_task_id}")
        detector = Session(api, task_id=args.detector_task_id)
    elif hasattr(args, 'detector_model') and args.detector_model:
        # Развертывание новой модели
        logger.info(f"Deploying detector model: {args.detector_model}")
        detector = api.nn.deploy(
            model=args.detector_model,
            device=getattr(args, 'detector_device', 'cuda:0'),
            **getattr(args, 'detector_settings', {})
        )
    else:
        raise ValueError("Either 'detector_task_id' or 'detector_model' must be specified in config")
    
    return detector, api


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
            try:
                uuid_key = figure.video_object.key() 
                if uuid_key not in uuid_to_int:
                    uuid_to_int[uuid_key] = next_int_id
                    next_int_id += 1
            
                track_id = uuid_to_int[uuid_key]
                
            except Exception as e:
                logger.warning(f"Error converting track_id: {e}")
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


def run_original_botsort(args, frames_list, frame_to_annotation):
    """Запуск оригинального BoTSORT трекера напрямую"""
    logger.info("Starting Original BoTSORT tracking...")
    
    tracker_orig = BoTSORT_ORIG(args, frame_rate=args.fps)
    results = []
    
    for frame_id, frame in enumerate(frames_list, 1):
        # Получаем аннотацию для кадра
        annotation = frame_to_annotation.get(frame_id - 1)
        if annotation is None:
            logger.warning(f"No annotation for frame {frame_id}")
            results.append((frame_id, [], [], []))
            continue
        
        # Конвертируем аннотацию в формат детекций для трекера
        dets = []
        for label in annotation.labels:
            geom = label.geometry
            x1, y1, x2, y2 = geom.left, geom.top, geom.right, geom.bottom
            score = 1.0
            
            # Поиск тега с confidence
            for tag in label.tags:
                if tag.name in ['confidence', 'score', 'conf']:
                    score = float(tag.value)
                    break
            
            # Получаем класс (предполагаем, что это person для MOT)
            cls_id = 0  # person
            dets.append([x1, y1, x2, y2, score, cls_id])
        
        dets = np.array(dets) if dets else np.zeros((0, 6), dtype=float)
        
        # Обновляем трекер
        targets = tracker_orig.update(dets.copy(), frame)
        
        # Собираем результаты
        tlbrs, ids, scores = [], [], []
        for t in targets:
            x, y, w, h = t.tlwh
            if w * h > 10:
                tlbrs.append((x, y, x + w, y + h))
                ids.append(t.track_id)
                scores.append(t.score)
        
        results.append((frame_id, tlbrs, ids, scores))
    
    return results


def run_supervisely_botsort(args, frames_list, frame_to_annotation):
    """Запуск Supervisely BoTTracker через обертку"""
    logger.info("Starting Supervisely BoTSORT tracking...")
    
    # Настройки трекера
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
        "device": getattr(args, 'device', 'cpu'),
        "device_id": getattr(args, 'device_id', '0'),
    }
    
    # Добавляем ReID параметры если включен
    if args.with_reid:
        tracker_settings.update({
            "reid_model": getattr(args, 'reid_model', 'osnet_reid'),
            "reid_weights": getattr(args, 'reid_weights', None),
            "fast_reid_config": getattr(args, 'fast_reid_config', None),
            "fast_reid_weights": getattr(args, 'fast_reid_weights', None),
        })
    
    # Инициализация трекера
    tracker = BoTTracker(settings=tracker_settings)
    
    # Размер кадра
    frame_shape = (frames_list[0].shape[0], frames_list[0].shape[1])
    
    # Запуск трекинга
    video_annotation = tracker.track(
        source=frames_list,
        frame_to_annotation=frame_to_annotation,
        frame_shape=frame_shape
    )
    
    # Извлечение результатов
    results = extract_tracks_from_video_annotation(video_annotation)
    return results


def load_frames_and_get_detections(args):
    """Загрузка кадров и получение детекций через Supervisely API"""
    frames_list = []
    frame_to_annotation = {}
    
    # Создание детектора
    detector, api = create_supervisely_detector(args)
    
    # Определяем источник кадров
    if osp.isdir(args.input):
        image_paths = get_image_list(args.input)
        logger.info(f"Found {len(image_paths)} images in directory: {args.input}")
        
        # Обработка изображений
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            frames_list.append(frame)
            
            # Получение детекций через Supervisely API
            try:
                annotation = detector.inference_image_path(img_path)
                frame_to_annotation[i] = annotation
            except Exception as e:
                logger.error(f"Error in detection for {img_path}: {e}")
                # Создаем пустую аннотацию
                frame_to_annotation[i] = Annotation(img_size=(frame.shape[0], frame.shape[1]))
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed frame {i+1}/{len(image_paths)}")
                if frame_to_annotation[i] is not None:
                    logger.info(f"  - Detections: {len(frame_to_annotation[i].labels)}")
    
    elif osp.isfile(args.input):
        # Обработка видео файла
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            logger.error(f"Could not open video: {args.input}")
            return None, None
            
        frame_id = 0
        temp_dir = "/tmp/tracking_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_list.append(frame)
            
            # Сохраняем кадр во временный файл для детекции
            temp_path = osp.join(temp_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(temp_path, frame)
            
            # Получение детекций
            try:
                annotation = detector.inference_image_path(temp_path)
                frame_to_annotation[frame_id] = annotation
            except Exception as e:
                logger.error(f"Error in detection for frame {frame_id}: {e}")
                frame_to_annotation[frame_id] = Annotation(img_size=(frame.shape[0], frame.shape[1]))
            
            # Удаляем временный файл
            os.remove(temp_path)
            
            frame_id += 1
            if frame_id % 50 == 0:
                logger.info(f"Processed frame {frame_id}")
                if frame_to_annotation[frame_id-1] is not None:
                    logger.info(f"  - Detections: {len(frame_to_annotation[frame_id-1].labels)}")
        
        cap.release()
        os.rmdir(temp_dir)
        logger.info(f"Total frames read from video: {frame_id}")
    
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return None, None
    
    return frames_list, frame_to_annotation


def main():
    args = parse_args()
    
    # Создание директории для результатов
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("Loading frames and getting detections via Supervisely API...")
    frames_list, frame_to_annotation = load_frames_and_get_detections(args)
    
    if frames_list is None:
        logger.error("Failed to load frames")
        return
    
    logger.info(f"Loaded {len(frames_list)} frames")
    
    # Подсчет общего количества детекций
    total_detections = sum(len(ann.labels) for ann in frame_to_annotation.values() if ann is not None)
    logger.info(f"Total detections across all frames: {total_detections}")
    
    # Запуск разных трекеров
    results = {}
    
    # 1. Оригинальный BoTSORT
    if getattr(args, 'run_original', True):
        logger.info("=" * 50)
        logger.info("Running Original BoTSORT...")
        start_time = time.time()
        try:
            results['original'] = run_original_botsort(args, frames_list, frame_to_annotation)
            logger.info(f"Original BoTSORT completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Original BoTSORT failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. Supervisely BoTTracker (с исправленным BaseTracker)
    if getattr(args, 'run_supervisely', True):
        logger.info("=" * 50)
        logger.info("Running Supervisely BoTTracker...")
        start_time = time.time()
        try:
            results['supervisely'] = run_supervisely_botsort(args, frames_list, frame_to_annotation)
            logger.info(f"Supervisely BoTTracker completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Supervisely BoTTracker failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Сохранение результатов
    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY:")
    
    for method_name, result_data in results.items():
        output_filename = f"{method_name}_botsort.txt"
        output_path = osp.join(args.save_dir, output_filename)
        write_results(output_path, result_data)
        
        # Статистика
        total_tracks = set()
        total_detections = 0
        for _, _, ids, _ in result_data:
            total_tracks.update(ids)
            total_detections += len(ids)
        
        logger.info(f"{method_name.upper()} Statistics:")
        logger.info(f"  - Total unique tracks: {len(total_tracks)}")
        logger.info(f"  - Total detections: {total_detections}")
        logger.info(f"  - Average detections per frame: {total_detections/len(result_data) if len(result_data) > 0 else 0:.2f}")
        logger.info(f"  - Output file: {output_path}")
    
    logger.info("All tracking methods completed successfully!")


if __name__ == "__main__":
    main()