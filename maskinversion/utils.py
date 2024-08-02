import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv2
from legrad import list_pretrained as list_pretrained_legrad


def list_pretrained():
    return list_pretrained_legrad()

def batched_dice_loss(input, target):
    bs = target.shape[0]
    smooth = 1.

    iflat = input.view(bs, -1)
    tflat = target.view(bs, -1)
    intersection = (iflat * tflat).sum(-1)

    individual_dice = 1 - ((2. * intersection + smooth) / (iflat.sum(-1) + tflat.sum(-1) + smooth))
    return individual_dice.mean()


def overlay_image_mask(image, mask, color: list =[30, 144, 255], alpha=0.6, title='', show=True):
    color = np.array(color)
    image_np = np.array(image)
    if image_np.shape[:2] != mask.shape[:2]:
        image_np = np.array(image.resize((mask.shape[1], mask.shape[0])))

    if color.max() > 1:
        color = color / 255
    colored_mask = np.expand_dims(mask, axis=-1) * np.expand_dims(color, axis=[0, 1])
    highlighted_image = np.expand_dims(mask, axis=-1) * (alpha * (image_np / 255.) + \
                          colored_mask) + (1 - np.expand_dims(mask, axis=-1)) * (image_np / 255.)
    highlighted_image = np.clip(highlighted_image, a_min=0, a_max=1)

    if show:
        plt.imshow(highlighted_image)
        plt.title(title)
        plt.tight_layout()
        plt.axis('off')
        plt.show()
    return highlighted_image

def overlay_image_expl_map(image, expl_map, alpha=0.6, title='', show=True):
    assert isinstance(image, Image.Image), f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}'
    W, H = expl_map.shape[-2:]
    image = image.resize((H, W))

    if expl_map.ndim > 3:
        expl_map = expl_map[0]
    if isinstance(expl_map, torch.Tensor):
        expl_map = expl_map.detach().cpu().numpy()

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    expl_map = (expl_map * 255).astype('uint8')
    heatmap = cv2.applyColorMap(expl_map, cv2.COLORMAP_JET)

    overlay = (1 - alpha) * img_cv + alpha * heatmap
    overlay = cv2.cvtColor(overlay.astype('uint8'), cv2.COLOR_BGR2RGB)
    if show:
        plt.imshow(overlay)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return overlay