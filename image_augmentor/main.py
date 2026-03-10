import os
import argparse
import yaml
import random
from typing import List, Dict, Any

from auglib import (
    list_images, load_image, save_image, ensure_rgb, resize_image,
    random_flip, random_rotate, random_crop_resize,
    hsv_adjust, to_grayscale, color_balance,
    gaussian_noise, salt_pepper_noise,
    mixup_pair, cutmix_pair, mosaic_images
)

OP_MAP = {
    "random_flip": random_flip,
    "random_rotate": random_rotate,
    "random_crop_resize": random_crop_resize,
    "hsv_adjust": hsv_adjust,
    "to_grayscale": to_grayscale,
    "color_balance": color_balance,
    "gaussian_noise": gaussian_noise,
    "salt_pepper_noise": salt_pepper_noise,
    "mixup": mixup_pair,
    "cutmix": cutmix_pair,
    "mosaic": mosaic_images,
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_other_image(files: List[str], cur_path: str) -> str:
    if not files:
        return cur_path
    candidates = [p for p in files if p != cur_path]
    return random.choice(candidates) if candidates else cur_path


def apply_pipeline_to_image(img, img_path, all_files, pipeline, output_size, out_dir, base_name):
    cur = img.copy()
    tags = []

    for step in pipeline:
        name = step.get("name")
        params = step.get("params", {}) or {}

        if name not in OP_MAP:
            print(f"[WARN] Unknown operator '{name}' -> skip")
            continue

        p_global = float(params.get("p", 1.0))
        if random.random() > p_global:
            continue

        func = OP_MAP[name]

        if name == "mixup":
            other_path = choose_other_image(all_files, img_path)
            other_img = load_image(other_path) or cur.copy()
            other_img = ensure_rgb(other_img)
            other_img = resize_image(other_img, output_size)

            cur = resize_image(ensure_rgb(cur), output_size)
            alpha = float(params.get("alpha", 0.4))
            mixed, (_, _, lam) = func(cur, None, other_img, None, alpha=alpha)
            cur = mixed
            tags.append(f"mixup_l{lam:.3f}")

        elif name == "cutmix":
            other_path = choose_other_image(all_files, img_path)
            other_img = load_image(other_path) or cur.copy()
            other_img = ensure_rgb(other_img)
            other_img = resize_image(other_img, output_size)

            cur = resize_image(ensure_rgb(cur), output_size)
            alpha = float(params.get("alpha", 1.0))
            mixed, (_, _, lam_adj) = func(cur, None, other_img, None, alpha=alpha)
            cur = mixed
            tags.append(f"cutmix_l{lam_adj:.3f}")

        elif name == "mosaic":
            picks = [img_path]
            if len(all_files) >= 4:
                others = [p for p in all_files if p != img_path]
                picks += random.sample(others, k=3) if len(others) >= 3 else [random.choice(all_files) for _ in range(3)]
            else:
                picks += [random.choice(all_files) for _ in range(3)]

            group = []
            for p in picks:
                im = load_image(p)
                if im is None:
                    im = cur.copy()
                im = ensure_rgb(im)
                im = resize_image(im, output_size)
                group.append(im)

            mosaic_img, _ = func(group, [None]*4, out_size=output_size)
            cur = mosaic_img
            tags.append("mosaic")

        else:
            cur = ensure_rgb(cur)
            call_params = params.copy()
            if "out_size" in call_params:
                pass
            else:
                if name == "random_crop_resize":
                    call_params["out_size"] = output_size
            try:
                cur = func(cur, **call_params)
            except TypeError:
                cur = func(cur)
            tags.append(name)


    cur = resize_image(ensure_rgb(cur), output_size)

    suffix = "_".join(tags) if tags else "orig"
    out_name = f"{base_name}_{suffix}.jpg"
    out_path = os.path.join(out_dir, out_name)
    save_image(out_path, cur)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--single_file", default=None, help="仅处理此文件")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = load_yaml(args.config)
    output_size = tuple(cfg.get("output_size", [224, 224]))
    pipeline = cfg.get("pipeline", [])

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = list_images(args.input_dir)
    if args.single_file:
        target = os.path.join(args.input_dir, args.single_file)
        if not os.path.exists(target):
            raise FileNotFoundError(f"single_file 指定的文件不存在: {target}")
        processing = [target]
    else:
        processing = all_files

    if not processing:
        print("[WARN] 未发现任何图片，检查 input_dir")
        return

    print(f"[INFO] config: {args.config}")
    print(f"[INFO] pipeline: {[s.get('name') for s in pipeline]}")
    print(f"[INFO] output_size: {output_size}")
    print(f"[INFO] processing {len(processing)} files -> {args.output_dir}")

    for path in processing:
        img = load_image(path)
        if img is None:
            print(f"[WARN] 读取失败, skip {path}")
            continue
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = apply_pipeline_to_image(img, path, all_files, pipeline, output_size, args.output_dir, base)
            print(f"[OK] {path} -> {out_path}")
        except Exception as e:
            print(f"[ERROR] 处理失败 {path}: {e}")


if __name__ == "__main__":
    main()