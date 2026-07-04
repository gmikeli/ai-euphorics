import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid


def load_and_resize_images(folder, device, size=(256, 256), clamp = True):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.PILToTensor(),
    ])

    images = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            images.append(transform(img).to(device))

    if clamp:
        images = [img.float() / 255.0 for img in images]

    return images

def save_image(tensor, folder = "outputs", filename = "image.png"):
    os.makedirs(folder, exist_ok=True)

    t = tensor.detach().cpu()

    if t.max() > 1.0:
        t = t / 255.0

    t = t.clamp(0.0, 1.0)
    t = (t * 255).byte()
    t = t.permute(1, 2, 0).numpy()

    image = Image.fromarray(t, mode="RGB")
    path = os.path.join(folder, filename)
    image.save(path)
    print(f"Saved: {path}")

def display_images(images):
    if type(images) is  list:
        batch = torch.stack(images).detach().to('cpu')
    else:
        batch = images

    grid = make_grid(batch, nrow=4)

    plt.close('all')
    # plt.figure(figsize=(10, 10))

    plt.imshow(grid.permute(1, 2, 0), vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


def pixels_to_pil(img_pixels):
    arr = np.asarray(img_pixels.clone().detach())
    if arr.shape[0] == 3:  # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def shuffle_image_dict(images):
    shuffled_image_dict = []
    for idx, img in enumerate(images):
        shuffled_image_dict.append({'original_idx': idx, 'img_pixels': img})

    random.shuffle(shuffled_image_dict)
    return shuffled_image_dict