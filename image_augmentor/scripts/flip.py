import argparse
import os
from auglib import list_images, load_image, save_image, ensure_rgb, random_flip, resize_image

def main(input_dir, output_dir, p, horizontal, vertical, out_size):
    os.makedirs(output_dir, exist_ok=True)
    for path in list_images(input_dir):
        img = load_image(path)
        if img is None:
            print("skip", path)
            continue
        img = ensure_rgb(img)
        # apply flip on original size to preserve detail
        aug = random_flip(img, horizontal=horizontal, vertical=vertical, p=p)
        # then resize to desired output
        aug = resize_image(aug, out_size)
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(os.path.join(output_dir, f"{base}_flip.jpg"), aug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--horizontal", action="store_true")
    parser.add_argument("--vertical", action="store_true")
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.p, args.horizontal, args.vertical, tuple(args.out_size))