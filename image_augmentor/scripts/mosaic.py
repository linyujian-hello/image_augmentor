import argparse, os
from auglib import list_images, load_image, save_image, ensure_rgb, resize_image, mosaic_images

def main(input_dir, output_dir, out_size):
    os.makedirs(output_dir, exist_ok=True)
    files = list_images(input_dir)
    n = len(files)
    if n == 0:
        return
    idx = 0
    count = 0
    while idx < n:
        group = []
        labels = []
        for k in range(4):
            path = files[(idx + k) % n]
            img = load_image(path)
            if img is None:
                img = load_image(files[0])
            img = ensure_rgb(img)
            img = resize_image(img, out_size)
            group.append(img)
            labels.append(None)
        mosaic, _ = mosaic_images(group, labels, out_size=out_size)
        save_image(os.path.join(output_dir, f"mosaic_{count:04d}.jpg"), mosaic)
        idx += 4
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, tuple(args.out_size))