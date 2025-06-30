import argparse
import motmetrics as mm
import pandas as pd

def compute_metrics(gt_path, pred_path, tracker_name):
    gt = mm.io.loadtxt(gt_path, fmt='mot15-2D', min_confidence=0)
    pred = mm.io.loadtxt(pred_path, fmt='mot15-2D')

    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(gt.index.get_level_values(0)) | set(pred.index.get_level_values(0)))

    for frame in frames:
        gt_frame = gt.xs(frame, level=0, drop_level=False) if frame in gt.index.get_level_values(0) else pd.DataFrame()
        pred_frame = pred.xs(frame, level=0, drop_level=False) if frame in pred.index.get_level_values(0) else pd.DataFrame()

        gt_ids = gt_frame.index.get_level_values(1).values if not gt_frame.empty else []
        pred_ids = pred_frame.index.get_level_values(1).values if not pred_frame.empty else []

        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values if not gt_frame.empty else []
        pred_boxes = pred_frame[['X', 'Y', 'Width', 'Height']].values if not pred_frame.empty else []

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    # summary = mh.compute(acc, metrics=[
    #     'num_frames', 'mota', 'idf1', 'idp', 'idr',
    #     'precision', 'recall', 'num_switches', 'fp', 'fn'
    # ], name=tracker_name)
    
    summary = mh.compute(acc, metrics=[
        'num_frames', 'idf1', 'idp', 'idr',
        'recall', 'precision', 'num_objects',
        'mostly_tracked', 'partially_tracked', 'mostly_lost',
        'num_false_positives', 'num_misses', 'num_switches',
        'mota', 'motp'
    ])



    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='Path to ground truth .txt file')
    parser.add_argument('--botsort', required=True, help='Path to BoT-SORT predictions')
    parser.add_argument('--botsort_reid', required=True, help='Path to BoT-SORT with ReID predictions')
    parser.add_argument('--boxmot', required=True, help='Path to BoxMOT predictions')
    parser.add_argument('--boxmot_reid', required=True, help='Path to BoxMOT with ReID predictions')
    parser.add_argument('--output', required=True, help='Path to save output metrics')
    args = parser.parse_args()
    
    res1 = compute_metrics(args.gt, args.botsort, 'botsort')
    res2 = compute_metrics(args.gt, args.botsort_reid, 'botsort_reid')
    res3 = compute_metrics(args.gt, args.boxmot, 'boxmot')
    res4 = compute_metrics(args.gt, args.boxmot_reid, 'boxmot_reid')

    combined = pd.concat([res1, res2, res3, res4])
    combined['tracker'] = ['botsort', 'botsort_reid', 'boxmot', 'boxmot_reid']
    print("Сравнительная таблица:\n")
    print(combined[['tracker', 'mota', 'idf1', 'precision', 'recall', 'num_switches', 'num_false_positives', 'num_misses']])
    combined.to_excel(args.output, index=False)
    print(f"Сохранено в {args.output}")

if __name__ == '__main__':
    main()
