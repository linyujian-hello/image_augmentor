import argparse, os
from auglib import list_images, load_image, save_image, ensure_rgb, resize_image, gaussian_noise, salt_pepper_noise

def main(input_dir, output_dir, p_gauss, var, p_sp, density, out_size):
    os.makedirs(output_dir, exist_ok=True)
    for path in list_images(input_dir):
        img = load_image(path)
        if img is None:
            continue
        img = ensure_rgb(img)
        img = resize_image(img, out_size)
        out = gaussian_noise(img, var=var, p=p_gauss)
        out = salt_pepper_noise(out, density=density, p=p_sp)
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(os.path.join(output_dir, f"{base}_noise.jpg"), out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p_gauss", type=float, default=0.5)
    parser.add_argument("--var", type=float, default=0.02)
    parser.add_argument("--p_sp", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=0.02)
    parser.add_argument("--out_size", type=int, nargs=2, default=[224,224])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.p_gauss, args.var, args.p_sp, args.density, tuple(args.out_size))