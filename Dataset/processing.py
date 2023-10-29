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

    compose_list = []
    if 'RandomHorizontalFlip' in cfg.Data.normalize.augmentation:
        compose_list.append(transforms.RandomHorizontalFlip())
    if 'CenterCrop' in cfg.Data.normalize.augmentation:
        compose_list.append(transforms.CenterCrop(cfg.Data.normalize.crop_size))

    compose_list.append(transforms.Resize(cfg.Data.image_shape[1], cfg.Data.image_shape[2]))
    compose_list.append(transforms.ToTensor())
    compose_list.append(transforms.Normalize(mean=cfg.Data.normalize.mean, std=cfg.Data.normalize.std))

    cfg.normalize = transforms.Compose(compose_list)

def style_transfer_process(image_path, transform=None, device='cpu'):
    image = Image.open(image_path)

    if transform:
        image = transform(image)

    return image.unsqueeze(0).to(device)
    