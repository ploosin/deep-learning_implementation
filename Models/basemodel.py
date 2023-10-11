from abc import ABCMeta, abstractmethod
import torch.nn as nn
import logging
import time

from utils.tqdm_logger import TqdmLoggingHandler

class BaseModel(nn.Module, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        log = logging.getLogger(__name__)
        log.handlers = []
        log.setLevel(logging.INFO)
        log.addHandler(TqdmLoggingHandler())

    @abstractmethod
    def set_loss(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def set_optimizer(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def create_required_directory(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    