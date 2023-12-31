import yaml
import argparse
from easydict import EasyDict
from multiprocessing import freeze_support

from Dataset.dataloader import BasicLoader
from Dataset.processing import set_image_processing
from Models.GAN.gan import GenerativeAdversarialNetworks
from Models.GAN.dcgan import DeConvolutionGenerativeAdversarialNetworks
from Models.StyleTransfer.styletransfer import StyleTransfer
from Models.ImageToTranslation.pix2pix import Pix2Pix
from Models.ImageToTranslation.starganv1 import StarGANv1

import torch

def main():
    parser = argparse.ArgumentParser(prog='DeepLearningTrainer')
    parser.add_argument('-c', '--cfg', type=str, default='configs/starganv1_celeba.yml')
    args = parser.parse_args()

    cfg = yaml.load(open(args.cfg, 'rb'), Loader=yaml.Loader)
    cfg = EasyDict(cfg)

    # 1. Device
    if (torch.cuda.is_available()):
        cfg.device = torch.device('cuda')
        torch.cuda.set_per_process_memory_fraction(0.7)
    else:
        cfg.device = torch.device('cpu')

    # 2. PreProcessing
    set_image_processing(cfg)

    # 2. DataLoader
    if cfg.Data.dataset:
        dataset = BasicLoader(cfg)
        train_loader, _ = dataset.train_loader, dataset.test_loader
    else:
        train_loader = None

    # 3. Model
    model = StarGANv1(cfg)
    # model = torch.compile(model)
    model.train(train_loader)

if __name__ == '__main__':
    freeze_support()
    main()