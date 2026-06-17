"""Partial-image generation for STNReID (paper §IV "Generate Partial Images").

We randomly pick a direction (top/bottom/left/right) and keep ``ratio in [min, max]`` of the
holistic image along that direction to simulate an occlusion. The returned PIL crop is then
resized/normalised by the dataset's deterministic transform so the partial image matches the
holistic image size (required by ``F.grid_sample`` in the STN).
"""

import random

from PIL import Image


def generate_partial_image(img: Image.Image, min_ratio: float = 0.2, max_ratio: float = 0.6, rng=None) -> Image.Image:
    """Crop a partial view keeping ``[min_ratio, max_ratio]`` of ``img`` in one direction."""
    rng = rng if rng is not None else random
    width, height = img.size
    ratio = rng.uniform(min_ratio, max_ratio)
    direction = rng.choice(["top", "bottom", "left", "right"])

    if direction == "top":
        new_h = max(1, int(round(height * ratio)))
        box = (0, 0, width, new_h)
    elif direction == "bottom":
        new_h = max(1, int(round(height * ratio)))
        box = (0, height - new_h, width, height)
    elif direction == "left":
        new_w = max(1, int(round(width * ratio)))
        box = (0, 0, new_w, height)
    else:  # right
        new_w = max(1, int(round(width * ratio)))
        box = (width - new_w, 0, width, height)

    return img.crop(box)


class PartialImageAugmentation:
    """Composable PIL->PIL transform wrapping :func:`generate_partial_image`."""

    def __init__(self, min_ratio: float = 0.2, max_ratio: float = 0.6):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        return generate_partial_image(img, self.min_ratio, self.max_ratio)
