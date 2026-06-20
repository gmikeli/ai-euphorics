import os

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid


def load_and_resize_images(folder, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    images = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            images.append(transform(img))

    return images

def display_images(images):
    batch = torch.stack(images)
    grid = make_grid(batch, nrow=4)

    plt.close('all')
    # plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    plt.show()