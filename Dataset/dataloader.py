import os
import random
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
        elif cfg.Data.dataset == 'facades':
            train_dataset = FacadesDataset(cfg, True)
            test_dataset  = FacadesDataset(cfg, False)
        elif cfg.Data.dataset == 'celeba':
            train_dataset = CelebADataset(cfg, True)
            test_dataset  = CelebADataset(cfg, False)
    
        self.train_loader = DataLoader(train_dataset,
                                        batch_size=cfg.Data.batch_size,
                                        num_workers=cfg.Data.num_workers,
                                        shuffle=True)
        self.test_loader = DataLoader(test_dataset,
                                        batch_size=cfg.Data.batch_size,
                                        num_workers=cfg.Data.num_workers,
                                        shuffle=False)
        
        print(f"Finish making DataLoader {len(train_dataset)} of train set and {len(test_dataset)} of test set")

class FacadesDataset(Dataset):      # 건물의 정면
    def __init__(self, cfg, train=True) -> None:
        super(FacadesDataset, self).__init__()

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
    

class CelebADataset(Dataset):
    def __init__(self, cfg, train=False) -> None:
        super(CelebADataset, self).__init__()

        self.cfg            = cfg
        self.image_dir      = cfg.Data.image_dir
        self.attr_path      = cfg.Data.attr_path
        self.selected_attrs = cfg.Data.selected_attrs
        self.normalize      = cfg.normalize
        self.train          = train
        self.train_dataset  = []
        self.test_dataset   = []
        self.attr2idx       = {}
        self.idx2attr       = {}
        self.preprocess()

        if train:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
    
    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        attr_names = lines[1].split()
        for i, attr_name in enumerate(attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        random.seed(0)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            fname = split[0]
            attrs = split[1:]
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(attrs[idx] == '1')
            if i + 1 < 2000:
                self.test_dataset.append([fname, label])
            else:
                self.train_dataset.append([fname, label])

    def __getitem__(self, index):
        dataset = self.train_dataset if self.train else self.test_dataset
        fname, label = dataset[index]
        img = Image.open(os.path.join(self.image_dir, fname))
        return self.normalize(img), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images
    