# Convert Supervisely video annotations to MOT format and compute metrics
import json
import pandas as pd
import numpy as np
import motmetrics as mm
import yaml
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments and load config file."""
    parser = argparse.ArgumentParser(description='Convert Supervisely annotations to MOT format and compute metrics')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_supervisely_annotation(json_path):
    """Load Supervisely video annotation from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def convert_sly_to_mot(sly_data, output_path, confidence_threshold=0.0):
    """
    Convert Supervisely annotation to MOT format.
    
    MOT format: frame_id, track_id, x, y, w, h, conf, x, y
    where x,y is top-left corner, w,h is width and height
    """
    mot_data = []
    
    # Create mapping from objectKey to track_id (starting from 0 like GT)
    object_keys = [obj['key'] for obj in sly_data['objects']]
    key_to_id = {key: idx for idx, key in enumerate(object_keys)}
    
    # Process each frame
    for frame in sly_data['frames']:
        frame_id = frame['index'] + 1  # MOT uses 1-based indexing
        
        for figure in frame['figures']:
            object_key = figure['objectKey']
            track_id = key_to_id[object_key]
            
            # Extract bounding box coordinates
            if figure['geometryType'] == 'rectangle':
                exterior_points = figure['geometry']['points']['exterior']
                x1, y1 = exterior_points[0]
                x2, y2 = exterior_points[1]
                
                # Convert to MOT format (top-left corner + width/height)
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                # MOT format: frame, id, x, y, w, h, conf, -1, -1
                mot_row = [frame_id, track_id, x, y, w, h, 1, 1, 1]
                mot_data.append(mot_row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(mot_data, columns=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x2', 'y2'])
    df = df.sort_values(['frame', 'id'])
    
    # Save in same format as GT: with spaces after commas
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            line = ', '.join([str(int(val)) for val in row])
            f.write(line + '\n')
    
    print(f"Converted annotation saved to: {output_path}")
    return df


def compute_hota_simple(gt_df, pred_df, iou_threshold=0.5):
    """
    Compute simplified HOTA metric.
    This is a basic approximation - for precise HOTA use TrackEval library.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        
        def bbox_iou(box1, box2):
            """Calculate IoU between two bounding boxes."""
            x1_min, y1_min, w1, h1 = box1
            x2_min, y2_min, w2, h2 = box2
            
            x1_max, y1_max = x1_min + w1, y1_min + h1
            x2_max, y2_max = x2_min + w2, y2_min + h2
            
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            union_area = w1 * h1 + w2 * h2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        total_hota = 0.0
        valid_frames = 0
        
        all_frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
        
        for frame_id in all_frames:
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            pred_frame = pred_df[pred_df['frame'] == frame_id]
            
            if gt_frame.empty or pred_frame.empty:
                continue
            
            # Calculate IoU matrix
            gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
            pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values
            
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou_matrix[i, j] = bbox_iou(gt_box, pred_box)
            
            # Simple HOTA approximation based on detection quality
            if iou_matrix.size > 0:
                max_ious = np.max(iou_matrix, axis=1)
                frame_hota = np.mean(max_ious[max_ious > iou_threshold])
                if not np.isnan(frame_hota):
                    total_hota += frame_hota
                    valid_frames += 1
        
        return total_hota / valid_frames if valid_frames > 0 else 0.0
        
    except ImportError:
        return None


def compute_mot_metrics(gt_path, pred_path):
    """Compute MOT metrics comparing ground truth and predictions."""
    try:
        # Load data with pandas (motmetrics has issues with space-separated files)
        gt_df = pd.read_csv(gt_path, header=None, 
                          names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x2', 'y2'])
        pred_df = pd.read_csv(pred_path, header=None,
                            names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x2', 'y2'])
        
        # Compute MOT metrics using motmetrics
        acc = mm.MOTAccumulator(auto_id=True)
        
        all_frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
        
        for frame_id in all_frames:
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            pred_frame = pred_df[pred_df['frame'] == frame_id]
            
            # Extract data
            gt_ids = gt_frame['id'].values if not gt_frame.empty else np.array([])
            pred_ids = pred_frame['id'].values if not pred_frame.empty else np.array([])
            
            # Convert to bbox format for IoU calculation
            gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values if not gt_frame.empty else np.empty((0, 4))
            pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values if not pred_frame.empty else np.empty((0, 4))
            
            # Calculate IoU distance matrix
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            
            # Update accumulator
            acc.update(gt_ids, pred_ids, distances)
        
        # Compute standard MOT metrics
        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                'num_frames', 'idf1', 'idp', 'idr',
                'recall', 'precision', 'num_objects',
                'mostly_tracked', 'partially_tracked', 'mostly_lost',
                'num_false_positives', 'num_misses', 'num_switches',
                'mota', 'motp'
            ]
        )
        
        # Add HOTA if possible
        hota_score = compute_hota_simple(gt_df, pred_df)
        if hota_score is not None:
            summary['hota'] = hota_score
        
        return summary
        
    except Exception as e:
        raise ValueError(f"Error computing metrics: {e}")


def main():
    config = parse_args()
    print("Starting Supervisely to MOT conversion and metrics computation...")
    
    # Convert Supervisely annotation to MOT format
    sly_data = load_supervisely_annotation(config['sly_annotation_path'])
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to MOT format
    converted_path = output_dir / config['converted_filename']
    mot_df = convert_sly_to_mot(
        sly_data, 
        converted_path, 
        confidence_threshold=config.get('confidence_threshold', 0.0)
    )
    
    print(f"Converted {len(mot_df)} detections from {len(sly_data['frames'])} frames")
    
    # Compute metrics if ground truth is provided
    if config.get('ground_truth_path'):
        print("Computing MOT metrics...")
        
        try:
            metrics = compute_mot_metrics(
                config['ground_truth_path'],
                converted_path
            )
            
            # Save metrics to Excel
            metrics_path = output_dir / config['metrics_filename']
            metrics.to_excel(metrics_path, index=True)
            print(f"Metrics saved to: {metrics_path}")
            
            # Print key metrics
            print("\nKey Metrics:")
            print(f"MOTA: {metrics.iloc[0]['mota']:.4f}")
            print(f"IDF1: {metrics.iloc[0]['idf1']:.4f}")
            print(f"Precision: {metrics.iloc[0]['precision']:.4f}")
            print(f"Recall: {metrics.iloc[0]['recall']:.4f}")
            print(f"ID Switches: {int(metrics.iloc[0]['num_switches'])}")
            
            # Print HOTA if available
            if 'hota' in metrics.columns:
                print(f"HOTA: {metrics.iloc[0]['hota']:.4f}")
            else:
                print("HOTA: Not available (install scipy for simplified HOTA)")
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
    
    else:
        print("No ground truth provided, skipping metrics computation")
    
    print("Done!")


if __name__ == "__main__":
    main()