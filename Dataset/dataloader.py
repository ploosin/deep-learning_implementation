import os
from PIL import Image
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

class BasicLoader(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        train_data_path = os.path.join(cfg.data_path, cfg.Data.dataset, 'train')
        test_data_path  = os.path.join(cfg.data_path, cfg.Data.dataset, 'test')

        if   cfg.Data.dataset == 'MNIST':
            train_dataset = datasets.MNIST(
                train_data_path,
                train=True,
                download=True,
                transform=cfg.normalize,
            )
            test_dataset = datasets.MNIST(
                test_data_path,
                train=False,
                download=True,
                transform=cfg.normalize,
            )
        elif cfg.Data.dataset == 'CIFAR10':
            train_dataset = datasets.CIFAR10(
                train_data_path,
                train=True,
                download=False,
                transform=cfg.normalize,
            )
            test_dataset = datasets.CIFAR10(
                test_data_path,
                train=False,
                download=False,
                transform=cfg.normalize,
            )
        elif cfg.Model.name == 'Pix2Pix':
            train_dataset = Pix2PixDataset(cfg, True)
            test_dataset  = Pix2PixDataset(cfg, False)
    
        self.train_loader = DataLoader(train_dataset,
                                        batch_size=cfg.Data.batch_size,
                                        num_workers=cfg.Data.num_workers,
                                        shuffle=True)
        self.test_loader = DataLoader(test_dataset,
                                        batch_size=cfg.Data.batch_size,
                                        num_workers=cfg.Data.num_workers,
                                        shuffle=False)

class Pix2PixDataset(Dataset):
    def __init__(self, cfg, train) -> None:
        super(Pix2PixDataset, self).__init__()

        self.cfg       = cfg
        self.direction = cfg.Data.direction     # a2b or b2a
        self.normalize = cfg.normalize
        if train:
            self.a_path    = cfg.Data.train.a_directory
            self.b_path    = cfg.Data.train.b_directory
        else:
            self.a_path    = cfg.Data.test.a_directory
            self.b_path    = cfg.Data.test.b_directory
        self.img_filenames = [x for x in os.listdir(self.a_path)]

    def __getitem__(self, index):
        a = Image.open(os.path.join(self.a_path, self.img_filenames[index])).convert('RGB')
        b = Image.open(os.path.join(self.b_path, self.img_filenames[index])).convert('RGB')

        a = self.normalize(a)
        b = self.normalize(b)

        if self.direction == 'a2b':
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.img_filenames)
    
    