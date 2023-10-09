import yaml
from easydict import EasyDict

from Dataset.dataloader import BasicDataset
from Models.gan import GenerativeAdversarialNetworks

import torch

def main():
    cfg = yaml.load(open('configs/gan_cifar.yml', 'rb'), Loader=yaml.Loader)
    cfg = EasyDict(cfg)

    # 1. Device
    if (torch.cuda.is_available()):
        cfg.device = torch.device('cuda')
        torch.cuda.set_per_process_memory_fraction(0.7)

    # 2. DataLoader
    dataset = BasicDataset(cfg)
    train_loader, _ = dataset.train_loader, dataset.test_loader

    # 3. Model
    model = GenerativeAdversarialNetworks(cfg)
    model.train(train_loader)

main()