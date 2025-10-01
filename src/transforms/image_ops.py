# 変換用関数

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

Array = np.ndarray
Size = Tuple[int, int]
def center_crop(img: Array, size: Union[int, Size]) -> Array:
    if isinstance(size, int):
        w = h = size
    else:
        w, h = size
    H, W = img.shape[:2]
    if w > W or h > H:
        return img
    x1 = (W - w) // 2
    y1 = (H - h) // 2
    return img[y1:y1+h, x1:x1+w]

def resize(img: Array, size: Size) -> Array:
    w, h = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def to_gray(img: Array) -> Array:
    if img.ndim == 2:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize(img: Array, mean=0.0, std=1.0) -> Array:
    x = img.astype(np.float32)/255.0
    if x.ndim == 3:
        m = np.array(mean if isinstance(mean, (list, tuple)) else (mean,)*3, dtype=np.float32)
        s = np.array(std if isinstance(std, (list, tuple)) else (std,)*3, dtype=np.float32)
        s = np.where(s==0, 1.0, s)
        return (x - m) / s
    
    s = std if std != 0 else 1.0
    return (x -(mean if not isinstance(mean, (list, tuple)) else mean[0])) / s

@dataclass(frozen=True)
class PreprocessConfig:
    target_size : Size = (224, 224)
    center_crop_size : Optional[Size] = None
    to_grayscale : bool = False
    mean : Union[float, Tuple[float, float, float]] = 0.0
    std : Union[float, Tuple[float, float, float]] = 1.0
    output_channels : int = 3  # 1 or 3

def preprocess_image(img: Array, cfg: PreprocessConfig) -> Array:
    x = img
    if cfg.center_crop_size is not None:
        x = center_crop(x, cfg.center_crop_size)
    x = resize(x, cfg.target_size)
    if cfg.to_grayscale:
        x = to_gray(x)
    x = normalize(x, mean = cfg.mean, std = cfg.std)
    if cfg.output_channels == 1 and x.ndim == 3:
        x = x.mean(axis=2, keepdims=True)
    elif cfg.output_channels == 3 and x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x