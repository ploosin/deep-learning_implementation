import yaml
from easydict import EasyDict

from Dataset.dataloader import BasicDataset
from Dataset.processing import set_image_processing, style_transfer_process
from Models.GAN.gan import GenerativeAdversarialNetworks
from Models.GAN.dcgan import DeConvolutionGenerativeAdversarialNetworks

import torch

def main():
    cfg = yaml.load(open('configs/dcgan_cifar.yml', 'rb'), Loader=yaml.Loader)
    cfg = EasyDict(cfg)

    # 1. Device
    if (torch.cuda.is_available()):
        cfg.device = torch.device('cuda')
        torch.cuda.set_per_process_memory_fraction(0.7)

    # 2. PreProcessing
    style_transfer_process(cfg)

    # 2. DataLoader
    dataset = BasicDataset(cfg)
    train_loader, _ = dataset.train_loader, dataset.test_loader

    # 3. Model
    model = DeConvolutionGenerativeAdversarialNetworks(cfg)
    model.train(train_loader)

main()