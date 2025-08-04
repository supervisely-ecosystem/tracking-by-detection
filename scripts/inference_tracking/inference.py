import os
import time
import json
from dotenv import load_dotenv
import supervisely as sly
from supervisely_integration.serve.serve_yolo import YOLOModel
from supervisely.nn.inference.inference_request import InferenceRequest, InferenceRequestsManager
import yaml
import argparse


def parse_args():
    """Parse command line arguments and load config file."""
    parser = argparse.ArgumentParser(description='Visualize Supervisely tracking annotations')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

config = parse_args()
        
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

# 1. create a model
model = YOLOModel(
    model="YOLOv8x-det (COCO)",
    device="cuda",
)

# 2. deploy model
model.serve()

# 3. inference with tracking
state = config['state']

video_path = config['video_path']
inference_request = InferenceRequest(manager=InferenceRequestsManager())

output = model._inference_video(path=video_path, state=state, inference_request=inference_request)

# 4. save results
with open(config['output_path'], "w") as f:
    json.dump(output, f)

