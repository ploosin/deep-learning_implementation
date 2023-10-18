import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import models, utils

from Models.basemodel import BaseModel
from Dataset.processing import style_transfer_process


class FeatureModel(nn.Module):
    def __init__(self) -> None:
        super(FeatureModel, self).__init__()

        self.model = models.vgg19(pretrained=True).features
        self.selected_layer = ['0', '5', '10', '19', '28']

    def _forward(self, x):
        features = []
        for n, layer in self.model._modules.items():
            x = layer(x)
            if n in self.selected_layer:
                features.append(x)
        return features
    
    def forward(self, x_list):
        features = []
        for x in x_list:
            features.append(self._forward(x))
        return features


class StyleTransfer(BaseModel):
    def __init__(self, cfg) -> None:
        super(StyleTransfer, self).__init__()

        self.cfg          = cfg
        self.device       = cfg.device
        
        self.model = FeatureModel().to(self.device)

        self.content_img = style_transfer_process(cfg.Data.content_img, cfg.normalize, self.device)
        self.style_img   = style_transfer_process(cfg.Data.style_img,   cfg.normalize, self.device)
        self.target_img  = self.content_img.clone().requires_grad_(True)

        self.set_loss()        
        self.set_optimizer()
        self.create_required_directory()

    def set_loss(self):
        self.criterion = nn.MSELoss()

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([self.target_img], lr= self.cfg.Optimizer.lr)
    
    def create_required_directory(self):
        self.save_image_dir = os.path.join(self.cfg.save_image_path, self.cfg.Model.name)
        self.save_model_dir = os.path.join(self.cfg.save_model_path, self.cfg.Model.name)
        os.makedirs(self.save_image_dir, exist_ok=True)
        os.makedirs(self.save_model_dir, exist_ok=True)

    def train(self, train_loader=None):
        for epoch in tqdm(range(self.cfg.epochs)):
            c, s, t = self.model([self.content_img, self.style_img, self.target_img])

            c_loss = 0
            s_loss = 0

            for c_feature, s_feature, t_feature in zip(c, s, t):
                c_loss += self.criterion(t_feature, c_feature)

                s_feature = s_feature.reshape(s_feature.shape[1], -1)
                t_feature = t_feature.reshape(t_feature.shape[1], -1)

                s_feature = torch.mm(s_feature, s_feature.t())
                t_feature = torch.mm(t_feature, t_feature.t())

                s_loss += self.criterion(t_feature, s_feature) / (s_feature.shape[0] * s_feature.shape[1])

            loss = c_loss + s_loss * self.cfg.Optimizer.style_loss_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % self.cfg.log_interval_epoch == 0:
                self.log.info(f'Epoch [{epoch+1}/{self.cfg.epochs}]  ContentLoss: {c_loss.item():.4f}  StyleLoss: {s_loss.item():.4f}')

            if (epoch+1) % self.cfg.save_interval_epoch == 0:
                save_image_path   = os.path.join(self.save_image_dir, 
                                                f'content_{self.cfg.Data.content_img.split("/")[-1]}_style_{self.cfg.Data.style_img.split("/")[-1]}_weight_{self.cfg.Optimizer.style_loss_weight}_epoch_{epoch+1}.png')
                
                save_img = self.target_img.squeeze(0).data.cpu()
                save_img = self.cfg.denormalize(save_img).clamp(0., 1.)
                utils.save_image(save_img, save_image_path)
