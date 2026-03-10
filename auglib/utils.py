import os
import cv2
import numpy as np
from typing import List, Optional, Tuple

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(SUPPORTED_EXT)]
    return files

def load_image(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def save_image(path: str, img_rgb: np.ndarray) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1] or ".jpg"
    success, buf = cv2.imencode(ext, bgr)
    if success:
        buf.tofile(path)

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    img = img.astype("uint8")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img

def resize_image(img: np.ndarray, out_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    h, w = out_size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)