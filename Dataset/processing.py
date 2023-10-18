import torch
from torchvision import transforms
from PIL import Image

def set_image_processing(cfg):
    if len(cfg.Data.normalize.mean) == 3:
        cfg.denormalize = lambda x: x*torch.tensor(
            [[[cfg.Data.normalize.std[0]]],  [[cfg.Data.normalize.std[1]]],  [[cfg.Data.normalize.std[2]]]]) + torch.tensor(
            [[[cfg.Data.normalize.mean[0]]], [[cfg.Data.normalize.mean[1]]], [[cfg.Data.normalize.mean[2]]]])
    else:
        cfg.denormalize = lambda x: x*cfg.Data.normalize.std + cfg.Data.normalize.mean

    cfg.normalize = transforms.Compose(
        [
            transforms.Resize((cfg.Data.image_shape[1], cfg.Data.image_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.Data.normalize.mean, std=cfg.Data.normalize.std)
        ]
    )

def style_transfer_process(image_path, transform=None, device='cpu'):
    image = Image.open(image_path)

    if transform:
        image = transform(image)

    return image.unsqueeze(0).to(device)
    