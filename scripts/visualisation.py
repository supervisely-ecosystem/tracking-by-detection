
import cv2, os
import imageio
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True, help='path to images')
parser.add_argument('--gt', required=True, help='path to ground thurts')
parser.add_argument('--botsort', required=True, help='path to botsort result')
parser.add_argument('--boxmot', required=True, help='path to boxmot result')
args = parser.parse_args()


def load_tracks(fn):
    df = pd.read_csv(fn, header=None, names=['frame','id','x','y','w','h','conf','a','b','c'])
    return df

os.makedirs('vis_frames', exist_ok=True)
gt = load_tracks(args.gt)
bot = load_tracks(args.botsort)
box = load_tracks(args.boxmot)
img_dir = args.images  # где лежат кадры
frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

# (255, 0, 0) → синий boxot
# (0, 255, 0) → зеленый GT
# (0, 0, 255) → красный botsort

colors = {'gt':(0,255,0),'bot':(0,0,255),'box':(255,0,0)}
images = []

for idx, fname in enumerate(frames, start=1):
    img = cv2.imread(os.path.join(img_dir,fname))
    for label, df, col in [('gt',gt,colors['gt']),('bot',bot,colors['bot']),('box',box,colors['box'])]:
        for _, row in df[df.frame==idx].iterrows():
            x,y,w,h=int(row.x),int(row.y),int(row.w),int(row.h)
            cv2.rectangle(img,(x,y),(x+w,y+h),col,2)
    outfn = f'vis_frames/{idx:04d}.png'
    cv2.imwrite(outfn,img)
    images.append(imageio.imread(outfn))

imageio.mimsave('comparison_yolox.gif', images, fps=5)
print("GIF saved as comparison.gif")
