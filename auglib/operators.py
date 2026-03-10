import numpy as np
import cv2
import random
from typing import Tuple, List


def random_flip(img: np.ndarray, horizontal=True, vertical=False, p=0.5) -> np.ndarray:

    if random.random() >= p:
        return img
    out = img.copy()
    # flip independently if both allowed
    if horizontal and random.random() < 0.5:
        out = np.flip(out, axis=1).copy()
    if vertical and random.random() < 0.5:
        out = np.flip(out, axis=0).copy()
    return out

def random_rotate(img: np.ndarray, min_angle=-30, max_angle=30, p=0.5, border_value=(0,0,0)) -> np.ndarray:

    if random.random() >= p:
        return img
    angle = random.uniform(min_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=border_value, flags=cv2.INTER_LINEAR)
    # warpAffine outputs same dtype
    return rotated

def random_crop_resize(img: np.ndarray, scale_min=0.8, scale_max=1.0, p=0.5, out_size=(224,224)) -> np.ndarray:

    if random.random() >= p:
        return cv2.resize(img, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    # 若图像非常小，避免 new_h/new_w = 0
    scale = random.uniform(scale_min, scale_max)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    if h - new_h > 0:
        top = random.randint(0, h - new_h)
    else:
        top = 0
    if w - new_w > 0:
        left = random.randint(0, w - new_w)
    else:
        left = 0
    crop = img[top:top+new_h, left:left+new_w]
    # 将裁剪区域缩放到 out_size
    return cv2.resize(crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)


def hsv_adjust(img: np.ndarray, hue_delta=30, sat_min=0.5, sat_max=1.5, val_min=0.5, val_max=1.5, p=0.8) -> np.ndarray:
    if random.random() >= p:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    dh = random.uniform(-hue_delta, hue_delta)
    # H 范围为 [0,179] in OpenCV; add dh then mod 180
    h = (h + dh) % 180.0
    s = s * random.uniform(sat_min, sat_max)
    v = v * random.uniform(val_min, val_max)
    # clip ranges: H [0,179], S/V [0,255]
    h = np.clip(h, 0, 179)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

def to_grayscale(img: np.ndarray, p=0.2) -> np.ndarray:
    if random.random() >= p:
        return img
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

def color_balance(img: np.ndarray, clip_limit=2.0, tile_grid=(8,8), p=0.5) -> np.ndarray:
    if random.random() >= p:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    merged = cv2.merge([cl, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# ----------------- 噪声 -----------------
def gaussian_noise(img: np.ndarray, mean=0.0, var=0.01, p=0.5) -> np.ndarray:

    if random.random() >= p:
        return img
    img_f = img.astype(np.float32) / 255.0
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, img_f.shape).astype(np.float32)
    out = np.clip(img_f + noise, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)

def salt_pepper_noise(img: np.ndarray, density=0.02, p=0.5) -> np.ndarray:

    if random.random() >= p:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    num = int(density * h * w)
    if num <= 0:
        return out
    # white (salt)
    xs_white = np.random.randint(0, h, num)
    ys_white = np.random.randint(0, w, num)
    out[xs_white, ys_white, :] = 255
    # black (pepper)
    xs_black = np.random.randint(0, h, num)
    ys_black = np.random.randint(0, w, num)
    out[xs_black, ys_black, :] = 0
    return out

# ----------------- 混合 / 拼接 -----------------
def mixup_pair(img1: np.ndarray, label1, img2: np.ndarray, label2, alpha=0.4) -> Tuple[np.ndarray, Tuple]:
    lam = np.random.beta(alpha, alpha)
    mixed = (img1.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam))
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    return mixed, (label1, label2, lam)

def cutmix_pair(img1: np.ndarray, label1, img2: np.ndarray, label2, alpha=1.0) -> Tuple[np.ndarray, Tuple]:
    h, w = img1.shape[:2]
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    new = img1.copy()
    # ensure indices form a valid box
    if x2 > x1 and y2 > y1:
        new[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    replaced = max(0, (x2 - x1)) * max(0, (y2 - y1))
    lam_adjusted = 1.0 - (replaced / (w * h)) if (w * h) > 0 else 1.0
    return new, (label1, label2, lam_adjusted)

def mosaic_images(imgs: List[np.ndarray], labels: List, out_size=(224,224)) -> Tuple[np.ndarray, List]:
    assert len(imgs) == 4, "mosaic requires 4 images"
    h_out, w_out = out_size
    h2, w2 = h_out // 2, w_out // 2
    parts = [cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA) for im in imgs]
    top = np.concatenate([parts[0], parts[1]], axis=1)
    bottom = np.concatenate([parts[2], parts[3]], axis=1)
    mosaic = np.concatenate([top, bottom], axis=0)
    return mosaic, labels