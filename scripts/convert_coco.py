import torch
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True, help='input path to YOLOX weights')
parser.add_argument('--output_path', required=True, help='output path to save converted weights')
args = parser.parse_args()


# Исходные веса от Megvii
ckpt = torch.load(args.input_path, map_location="cpu")

new_ckpt = OrderedDict()

for k, v in ckpt["model"].items():
    if k.startswith("backbone."):
        # Одинарное добавление
        new_key = "backbone.backbone." + k[len("backbone."):]
    else:
        new_key = k
    new_ckpt[new_key] = v

torch.save({"model": new_ckpt}, args.output_path)
