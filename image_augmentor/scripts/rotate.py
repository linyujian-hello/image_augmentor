import argparse
import os
from auglib import list_images, load_image, save_image, ensure_rgb, random_rotate, resize_image

def main(input_dir, output_dir, p, min_angle, max_angle, out_size):
    os.makedirs(output_dir, exist_ok=True)
    for path in list_images(input_dir):
        img = load_image(path)
        if img is None:
            print("skip", path)
            continue
        img = ensure_rgb(img)
        # apply rotation on original size
        aug = random_rotate(img, min_angle=min_angle, max_angle=max_angle, p=p)
        # then resize to desired output
        aug = resize_image(aug, out_size)
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(os.path.join(output_dir, f"{base}_rot.jpg"), aug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--min_angle", type=float, default=-30.0)
    parser.add_argument("--max_angle", type=float, default=30.0)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.p, args.min_angle, args.max_angle, tuple(args.out_size))