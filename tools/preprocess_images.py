import os
import argparse
from PIL import Image

def preprocess(input_dir, output_dir, out_size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
            continue
        src = os.path.join(input_dir, fname)
        try:
            im = Image.open(src).convert("RGB")
            im = im.resize(out_size, Image.BILINEAR)
            im.save(os.path.join(output_dir, fname))
        except Exception as e:
            print("skip damaged file:", src, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    preprocess(args.input_dir, args.output_dir, tuple(args.out_size))