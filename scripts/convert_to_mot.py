import json
import os
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', required=True, help='path to input JSON file with labels')
parser.add_argument('--out_path', required=True, help='path to save output MOT format file')
args = parser.parse_args()

json_path = args.json_path
out_path = args.out_path
os.makedirs('gt', exist_ok=True)

with open(json_path) as f:
    data = json.load(f)

frame_dict = {}
for item in data:
    frame = item['frameIndex'] + 1  # MOT starts at 1
    name = item['name']
    for label in item['labels']:
        # if label['category'] not in ['person', 'pedestrian']:
        #     continue
        box = label['box2d']
        x1, y1 = box['x1'], box['y1']
        w = box['x2'] - box['x1']
        h = box['y2'] - box['y1']
        track_id = int(label['id'][-6:], 16) % 100000  # generate some ID
        line = f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
        frame_dict.setdefault(frame, []).append(line)

with open(out_path, 'w') as out:
    for frame in sorted(frame_dict.keys()):
        for line in frame_dict[frame]:
            out.write(line)
