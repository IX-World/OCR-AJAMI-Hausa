"""IX-World: original fixed eight-view and V9 dynamic augmentation recipes."""
import random
from PIL import Image, ImageEnhance, ImageFilter
SEED = 42

def fixed_augmentation(image, variant, sample_index):
    if variant == 0:
        return image
    rng = random.Random(SEED + sample_index * 10007 + variant * 1000003)
    original_w, original_h = image.size
    if variant == 1:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.1))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.18))
        return image
    if variant == 2:
        if rng.random() < 0.5:
            image = image.filter(ImageFilter.MinFilter(3))
        else:
            image = image.filter(ImageFilter.MaxFilter(3))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.1))
        return image
    if variant == 3:
        scale = rng.uniform(0.72, 0.9)
        w2 = max(8, round(original_w * scale))
        h2 = max(8, round(original_h * scale))
        image = image.resize((w2, h2), Image.Resampling.BILINEAR)
        image = image.resize((original_w, original_h), Image.Resampling.BICUBIC)
        return image
    if variant == 4:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.55)))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.94, 1.08))
        return image
    if variant == 5:
        angle = rng.uniform(-1.0, 1.0)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        return image
    if variant == 6:
        sx = rng.uniform(0.94, 1.06)
        sy = rng.uniform(0.96, 1.04)
        new_w = max(8, round(original_w * sx))
        new_h = max(8, round(original_h * sy))
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image
    if variant == 7:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.07))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.14))
        if rng.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.12, 0.35)))
        angle = rng.uniform(-0.6, 0.6)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        return image
    return image

def augment_image(image, rng):
    if rng.random() < 0.35:
        factor = rng.uniform(0.88, 1.12)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if rng.random() < 0.4:
        factor = rng.uniform(0.85, 1.18)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if rng.random() < 0.15:
        angle = rng.uniform(-0.75, 0.75)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    r = rng.random()
    if r < 0.035:
        image = image.filter(ImageFilter.MinFilter(3))
    elif r < 0.07:
        image = image.filter(ImageFilter.MaxFilter(3))
    if rng.random() < 0.08:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.12, 0.4)))
    if rng.random() < 0.08:
        w, h = image.size
        scale = rng.uniform(0.82, 0.95)
        low_w = max(8, round(w * scale))
        low_h = max(8, round(h * scale))
        image = image.resize((low_w, low_h), Image.Resampling.BILINEAR)
        image = image.resize((w, h), Image.Resampling.BICUBIC)
    return image
