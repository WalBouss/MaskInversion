import requests
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import create_model, get_tokenizer, create_model_and_transforms

from maskinversion_release.maskinversion import MaskInversion, MaskInversionImagePreprocess, MaskInversionMaskPreprocess
from maskinversion_release.utils import overlay_image_mask, overlay_image_expl_map


# ------ MaskInversion Hyperparameters (Optional) ------
lr = 0.5
alpha = 0.
wd = 0.
optimizer = torch.optim.AdamW
# ------ init model ------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained = 'openai'
model_name = 'ViT-B-16'
model, _, preprocess = create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
tokenizer = get_tokenizer(model_name=model_name)

# ------ use MaskInversion wrapper ------
model = MaskInversion(model, lr=lr, alpha=alpha, wd=wd, optimizer=optimizer, iterations=10)
preprocess = MaskInversionImagePreprocess(preprocess, image_size=448)
mask_preprocess = MaskInversionMaskPreprocess()

# ------ init inputs ------
# === image ===
url_img = "https://github.com/WalBouss/Zero-shot-RIS/blob/main/assests/cats-and-dogs.jpg"
img_pil = Image.open(requests.get(url_img, stream=True).raw).convert('RGB')
image = preprocess(img_pil).unsqueeze(0).to(device)

# === masks ===
masks_urls = ['https://github.com/WalBouss/Zero-shot-RIS/blob/main/assests/cat.png',
              'https://github.com/WalBouss/Zero-shot-RIS/blob/main/assests/dog.png']
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
    image=image, masks_target=masks, verbose=True, return_expl_map=True
)  # [num_masks, dim]
localized_embeddings = F.normalize(localized_embeddings, dim=-1)

# ------ Region-Text matching ------
mask_text_matching = localized_embeddings @ text_embeddings.transpose(-1, -2) # [num_masks, num_prompt]
for i, mask in enumerate(masks.cpu().numpy()):
    print(f'{prompts[i]}: {mask_text_matching[i].softmax(dim=-1)}')
    matched_prompt_idx = mask_text_matching[i].argmax()

    # ___ (Optional): Visualize overlay of the image + mask ___
    overlay_image_mask(image=img_pil, mask=mask, show=True, title=prompts[matched_prompt_idx])
    # ___ (Optional): Visualize overlay of the image + heatmap ___
    overlay_image_expl_map(image=img_pil, expl_map=expl_map[0, i], title=prompts[matched_prompt_idx], show=True)
