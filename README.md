# MaskInversion
### [MaskInversion: Localized Embeddings via Optimization of Explainability Maps](https://arxiv.org/abs/2407.20034)
_[Walid Bousselham](http://walidbousselham.com/), [Sofian Chaybouti](https://scholar.google.com/citations?user=8tewdk4AAAAJ&hl),[Christian Rupprecht](https://chrirupp.github.io/), [Vittorio Ferrari](https://sites.google.com/view/vittoferrari), [Hilde Kuehne](https://hildekuehne.github.io/)_

The proposed method, coined as MaskInversion, aims to learn a localized embedding or feature vector that encapsulates an object’s characteristics within an image specified by a query mask. This embedding should not solely represent the object’s intrinsic properties but also capture the broader context of the entire image.

To achieve this, we utilize representations provided by foundation models, such as CLIP. Our approach learns a token that captures the foundation model’s feature representation on the image region specified by the mask. Hence, the foundation model remains fixed during our process.

The following is the code for a wrapper around the [OpenCLIP](https://github.com/mlfoundations/open_clip) library to equip VL models with the ability to compute "localized embeddings" via the MaskInversion process.

<div align="center">
<img src="./assets/maskinversion_teaser.png" width="100%"/>
</div>

## :hammer: Installation
`maskinversion` library can be simply installed via pip: 
```bash
$ pip install maskinversion_torch
```

## :firecracker: Usage

### Available models
MaskInversion uses the [LeGrad](https://github.com/WalBouss/LeGrad) library to compute the explainability maps, hence MaskInversion support all the models from that library.
To see which pretrained models is available use the following code snippet:
```python
import maskinversion
maskinversion.available_models()
```

### Example
Given an image and several masks covering different objects, you can run `python example_maskinversion.py` or use the following code snippet to compute the **localized embedding** for each mask:

**Note**: the wrapper does not affect the original model, hence all the functionalities of OpenCLIP models can be used seamlessly.
```python
import requests
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer, create_model_and_transforms
from maskinversion import (
 MaskInversion, MaskInversionImagePreprocess, MaskInversionMaskPreprocess, overlay_image_mask)

# ------ init model ------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained = 'openai'
model_name = 'ViT-B-16'
model, _, preprocess = create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
tokenizer = get_tokenizer(model_name=model_name)

# ------ use MaskInversion wrapper ------
model = MaskInversion(model)
preprocess = MaskInversionImagePreprocess(preprocess, image_size=448)
mask_preprocess = MaskInversionMaskPreprocess()

# ------ init inputs ------
# === image ===
url_img = "https://github.com/WalBouss/MaskInversion/blob/main/assests/dress_and_flower.png"
img_pil = Image.open(requests.get(url_img, stream=True).raw).convert('RGB')
image = preprocess(img_pil).unsqueeze(0).to(device)

# === masks ===
masks_urls = ['https://github.com/WalBouss/MaskInversion/blob/main/assests/dress_mask.png',
              'https://github.com/WalBouss/MaskInversion/blob/main/assests/flower_mask.png']
masks = [Image.open(requests.get(url, stream=True).raw) for url in masks_urls]
masks = torch.stack([mask_preprocess(msk) for msk in masks]).to(device)

# === text ===
prompts = ['a photo of a dress', 'a photo of a flower']
text_input = tokenizer(prompts).to(device)
text_embeddings = model.encode_text(text_input)  # [num_prompts, dim]
text_embeddings = F.normalize(text_embeddings, dim=-1)

# ------ Compute localized embedding for each mask ------
localized_embeddings = model.compute_maskinversion(image=image, masks_target=masks, verbose=True)  # [num_masks, dim]
localized_embeddings = F.normalize(localized_embeddings, dim=-1)

# ------ Region-Text matching ------
mask_text_matching = localized_embeddings @ text_embeddings.transpose(-1, -2) # [num_masks, num_prompt]
for i, mask in enumerate(masks.cpu().numpy()):
    print(f'{prompts[i]}: {mask_text_matching[i].softmax(dim=-1)}')
    matched_prompt_idx = mask_text_matching[i].argmax()

    # ___ (Optional): Visualize overlay of the image + mask ___
    overlay_image_mask(image=img_pil, mask=mask, show=True, title=prompts[matched_prompt_idx])
```
 
### Visualize the final Explainability Maps
To visualize the explainability map after the MaskInversion process you can run `python example_viz_expl_maps.py` or use the following code snippet:
```python
import requests
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer, create_model_and_transforms
from maskinversion import (
 MaskInversion, MaskInversionImagePreprocess, MaskInversionMaskPreprocess, overlay_image_expl_map)

# ------ init model ------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained = 'openai'
model_name = 'ViT-B-16'
model, _, preprocess = create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
tokenizer = get_tokenizer(model_name=model_name)

# ------ use MaskInversion wrapper ------
model = MaskInversion(model)
preprocess = MaskInversionImagePreprocess(preprocess, image_size=448)
mask_preprocess = MaskInversionMaskPreprocess()

# ------ init inputs ------
# === image ===
url_img = "https://github.com/WalBouss/MaskInversion/blob/main/assests/cats-and-dogs.jpg"
img_pil = Image.open(requests.get(url_img, stream=True).raw).convert('RGB')
image = preprocess(img_pil).unsqueeze(0).to(device)

# === masks ===
masks_urls = ['https://github.com/WalBouss/MaskInversion/blob/main/assests/dress_mask.png',
              'https://github.com/WalBouss/MaskInversion/blob/main/assests/flower_mask.png']
masks = [Image.open(requests.get(url, stream=True).raw) for url in masks_urls]
masks = torch.stack([mask_preprocess(msk) for msk in masks]).to(device)

# === text ===
prompts = ['a photo of a dress', 'a photo of a flower']
prompts = ['a photo of a cat', 'a photo of a dog']
text_input = tokenizer(prompts).to(device)
text_embeddings = model.encode_text(text_input)  # # [num_prompts, dim]
text_embeddings = F.normalize(text_embeddings, dim=-1)

# ------ Compute localized embedding for each mask ------
localized_embeddings, expl_map = model.compute_maskinversion(
 image=image, masks_target=masks, verbose=True, return_expl_map=True)  # [num_masks, dim]
localized_embeddings = F.normalize(localized_embeddings, dim=-1)

# ------ Region-Text matching ------
mask_text_matching = localized_embeddings @ text_embeddings.transpose(-1, -2) # [num_masks, num_prompt]
for i, mask in enumerate(masks.cpu().numpy()):
    print(f'{prompts[i]}: {mask_text_matching[i].softmax(dim=-1)}')
    matched_prompt_idx = mask_text_matching[i].argmax()

    # ___ (Optional): Visualize overlay of the image + heatmap ___
    overlay_image_expl_map(image=img_pil, expl_map=expl_map[0, i], title=prompts[matched_prompt_idx], show=True)
```
### MaskInversion Hyperparameters
You can manually set the different hyperparameters used for the MaskInversion process,
_e.g._ number of `iterations`, learning rate (`lr`), the optimizer use (`optimizer`), weight decay (`wd`) or the coefficient `alpha` for the regularization loss.
```python
iterations = 10
lr = 0.5
alpha = 0.
wd = 0.
optimizer = torch.optim.AdamW
model = MaskInversion(model=model, iterations=iterations, lr=lr, alpha=alpha, wd=wd, optimizer=optimizer)
```

# :star: Acknowledgement
This code is build as wrapper around [OpenCLIP](https://github.com/mlfoundations/open_clip) library from [LAION](https://laion.ai/) and the [LeGrad](https://github.com/WalBouss/LeGrad) library, visit their repo for more vision-language models.
This project also takes inspiration from [AlphaCLIP](https://github.com/SunzeY/AlphaCLIP) and the [timm library](https://github.com/huggingface/pytorch-image-models), please visit their repository.

# :books: Citation
If you find this repository useful, please consider citing our work :pencil: and giving a star :star2: :
```
@article{bousselham2024maskinversion,
  title={MaskInversion: Localized Embeddings via Optimization of Explainability Maps},
  author={Walid Bousselham, Sofian Chaybouti, Christian Rupprecht, Vittorio Ferrari, Hilde Kuehne},
  journal={arXiv preprint arXiv:2407.20034},
  year={2024}
}