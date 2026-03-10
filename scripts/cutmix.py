import argparse, os
from auglib import list_images, load_image, save_image, ensure_rgb, resize_image, cutmix_pair

def main(input_dir, output_dir, alpha, out_size):
    os.makedirs(output_dir, exist_ok=True)
    files = list_images(input_dir)
    n = len(files)
    for i in range(0, n, 2):
        if i+1 >= n:
            img = load_image(files[i])
            if img is None:
                continue
            img = ensure_rgb(img)
            img = resize_image(img, out_size)
            base = os.path.splitext(os.path.basename(files[i]))[0]
            save_image(os.path.join(output_dir, f"{base}_cutmix_single.jpg"), img)
            continue
        img1 = load_image(files[i])
        img2 = load_image(files[i+1])
        if img1 is None or img2 is None:
            continue
        img1 = ensure_rgb(img1)
        img2 = ensure_rgb(img2)
        img1 = resize_image(img1, out_size)
        img2 = resize_image(img2, out_size)
        mixed, (l1, l2, lam_adj) = cutmix_pair(img1, None, img2, None, alpha=alpha)
        base1 = os.path.splitext(os.path.basename(files[i]))[0]
        base2 = os.path.splitext(os.path.basename(files[i+1]))[0]
        save_image(os.path.join(output_dir, f"{base1}_{base2}_cutmix_{lam_adj:.3f}.jpg"), mixed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.alpha, tuple(args.out_size))