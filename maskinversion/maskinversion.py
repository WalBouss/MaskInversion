from legrad import LeWrapper, LePreprocess
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from .utils import batched_dice_loss


class MaskInversion(LeWrapper):
    def __init__(self, model, layer_index=-1, alpha=0., lr=0.1, iterations=10, wd=0., optimizer=optim.AdamW):
        LeWrapper.__init__(self, model, layer_index)
        self.float()
        self.optimizer = optimizer
        self.iterations = iterations
        self.lr = lr
        self.wd = wd
        self.alpha = alpha

    def compute_maskinversion(self, image, masks_target, alpha=None, lr=None, iterations=None, wd=None,
                              verbose=False, return_expl_map=False):
        """
        Computes mask inversion by optimizing mask embeddings to match target masks.

    This function performs an iterative optimization process to find mask embeddings
    that, when passed through the model, produce explainability maps closely matching
    the target masks. It uses a combination of mask loss and regularization loss.

    Args:
        image (torch.Tensor): Input image tensor of shape [1, C, H, W].
        masks_target (torch.Tensor): Target masks tensor of shape [N, H, W], where N is the number of masks.
        alpha (float, optional): Coefficient for regularization loss. If provided, updates the instance attribute.
        lr (float, optional): Learning rate for the optimizer. If provided, updates the instance attribute.
        iterations (int, optional): Number of optimization iterations. If provided, updates the instance attribute.
        wd (float, optional): Weight decay for the optimizer. If provided, updates the instance attribute.
        return_loss (bool, optional): If True, returns the final loss values along with the mask embeddings.

    Returns:
        torch.Tensor: Optimized mask embeddings of shape [N, num_patches, embedding_dim].
        dict (optional): Dictionary containing final loss values if return_loss is True.

    The function performs the following steps:
    1. Updates hyperparameters if new values are provided.
    2. Encodes the input image to obtain image features.
    3. Initializes mask embeddings based on the image features.
    4. Performs iterative optimization to refine the mask embeddings.
    5. Computes explainability maps from the mask embeddings.
    6. Calculates mask loss (dice loss) and regularization loss.
    7. Updates mask embeddings using the optimizer.

    The optimization process aims to minimize the combination of mask loss
    (difference between predicted and target masks) and regularization loss
    (to keep mask embeddings close to the original image features).
        """

        # -------- Update Hyperparameters --------
        if alpha is not None:
            self.alpha = alpha
        if lr is not None:
            self.lr = lr
        if iterations is not None:
            self.iterations = iterations
        if wd is not None:
            self.wd = wd

        # -------- Forward images --------
        num_masks = masks_target.shape[0]
        images = image.repeat(num_masks, 1, 1, 1)

        image_features = self.encode_image(images)  # [bs, num_patch, dim] bs=num_masks

        # -------- Init mask embeddings --------
        mask_emb = image_features.detach().clone()
        mask_emb = mask_emb.requires_grad_(True)
        # -------- init optimizer --------
        optimizer = self.optimizer([mask_emb], lr=self.lr, weight_decay=self.wd)

        # -------- Optimization steps --------
        start = time.time()
        for it in range(self.iterations):
            self.zero_grad()
            # --- get explainability map ---
            _mask_emb = F.normalize(mask_emb, dim=-1)
            expl_map = self.compute_legrad(text_embedding=_mask_emb)  # [b, num_maks, W, H]
            expl_map = F.interpolate(expl_map, size=masks_target.shape[-2:], mode='bilinear')

            # --- losses ---
            loss_reg = (1 - (F.normalize(image_features, dim=-1) * _mask_emb).sum(dim=-1)).mean()
            loss_mask = batched_dice_loss(input=expl_map, target=masks_target)
            loss = loss_mask + self.alpha * loss_reg
            if verbose:
                print(f'iteration {it}| loss: {loss.item():#.3G}')

            # --- optimization step ---
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        if verbose:
            print(f'total mask inversion time: {time.time() - start:#.4G} s')
        if return_expl_map:
            return mask_emb, expl_map
        return mask_emb


# ------ for easy import ------
MaskInversionImagePreprocess = LePreprocess

# ------ Mask preprocess ------
class MaskInversionMaskPreprocess(nn.Module):
    def __init__(self):
        super(MaskInversionMaskPreprocess, self).__init__()

    def forward(self, mask):
        return torch.as_tensor(np.array(mask) / 255., dtype=torch.float)
