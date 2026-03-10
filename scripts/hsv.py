import argparse, os
from auglib import list_images, load_image, save_image, ensure_rgb, resize_image, hsv_adjust, to_grayscale, color_balance

def main(input_dir, output_dir, p_hsv, hue_delta, sat_min, sat_max, val_min, val_max, p_gray, p_clahe, out_size):
    os.makedirs(output_dir, exist_ok=True)
    for path in list_images(input_dir):
        img = load_image(path)
        if img is None:
            continue
        img = ensure_rgb(img)
        img = resize_image(img, out_size)
        out = hsv_adjust(img, hue_delta=hue_delta, sat_min=sat_min, sat_max=sat_max, val_min=val_min, val_max=val_max, p=p_hsv)
        out = to_grayscale(out, p=p_gray)
        out = color_balance(out, p=p_clahe)
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(os.path.join(output_dir, f"{base}_hsv.jpg"), out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p_hsv", type=float, default=0.8)
    parser.add_argument("--hue_delta", type=float, default=30)
    parser.add_argument("--sat_min", type=float, default=0.6)
    parser.add_argument("--sat_max", type=float, default=1.4)
    parser.add_argument("--val_min", type=float, default=0.6)
    parser.add_argument("--val_max", type=float, default=1.4)
    parser.add_argument("--p_gray", type=float, default=0.1)
    parser.add_argument("--p_clahe", type=float, default=0.3)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.p_hsv, args.hue_delta, args.sat_min, args.sat_max, args.val_min, args.val_max, args.p_gray, args.p_clahe, tuple(args.out_size))