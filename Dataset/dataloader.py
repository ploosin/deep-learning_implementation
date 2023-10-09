import os
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BasicDataset(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        train_data_path = os.path.join(cfg.data_path, cfg.Data.dataset, 'train')
        test_data_path  = os.path.join(cfg.data_path, cfg.Data.dataset, 'test')
        cfg.denormalize = lambda x: x*0.5+0.5
        cfg.normalize = transforms.Compose(
            [
                transforms.Resize((cfg.Data.image_shape[1], cfg.Data.image_shape[2])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

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
    
        self.train_loader = DataLoader(train_dataset,
                                        batch_size=cfg.Data.batch_size,
                                        shuffle=True)
        self.test_loader = DataLoader(test_dataset,
                                        batch_size=cfg.Data.batch_size,
                                        shuffle=False)