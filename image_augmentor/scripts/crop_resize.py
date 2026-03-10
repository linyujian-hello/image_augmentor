import argparse
import os
from auglib import list_images, load_image, save_image, ensure_rgb, random_crop_resize

def main(input_dir, output_dir, p, scale_min, scale_max, out_size):
    os.makedirs(output_dir, exist_ok=True)
    for path in list_images(input_dir):
        img = load_image(path)
        if img is None:
            continue
        img = ensure_rgb(img)
        # 直接在原始尺寸上做随机裁剪并缩放到 out_size（保持多样性）
        out = random_crop_resize(img, scale_min=scale_min, scale_max=scale_max, p=p, out_size=out_size)
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(os.path.join(output_dir, f"{base}_crop.jpg"), out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p", type=float, default=0.6)
    parser.add_argument("--scale_min", type=float, default=0.8)
    parser.add_argument("--scale_max", type=float, default=1.0)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.p, args.scale_min, args.scale_max, tuple(args.out_size))