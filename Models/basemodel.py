from abc import ABCMeta, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        pass

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
