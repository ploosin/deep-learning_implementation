import os
import torch
import torch.nn as nn
import torchvision as tcvs
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from Models.basemodel import BaseModel


class Generator(nn.Module):
    def __init__(self, latent_size, feature_channel, output_channel):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, feature_channel * 8, 4, 1, 0, bias=False),          # [B x feature_channel*8 x 4 x 4]
            nn.BatchNorm2d(feature_channel * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_channel * 8, feature_channel * 4, 4, 2, 1, bias=False),  # [B x feature_channel*4 x 8 x 8]
            nn.BatchNorm2d(feature_channel * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( feature_channel * 4, feature_channel * 2, 4, 2, 1, bias=False), # [B x feature_channel*2 x 16 x 16]
            nn.BatchNorm2d(feature_channel * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( feature_channel * 2, feature_channel, 4, 2, 1, bias=False),     # [B x feature_channel*1 x 32 x 32]
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d( feature_channel, output_channel, 4, 2, 1, bias=False),          # [B x output_channel x 64 x 64]
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_channel, feature_channel):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(                                                     # input : [B x input_channel x 64 x 64]
            nn.Conv2d(input_channel, feature_channel, 4, 2, 1, bias=False),             # [B x feature_channel*1 x 32 x 32]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel, feature_channel * 2, 4, 2, 1, bias=False),       # [B x feature_channel*2 x 16 x 16]
            nn.BatchNorm2d(feature_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel * 2, feature_channel * 4, 4, 2, 1, bias=False),   # [B x feature_channel*4 x 8 x 8]
            nn.BatchNorm2d(feature_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel * 4, feature_channel * 8, 4, 2, 1, bias=False),   # [B x feature_channel*8 x 4 x 4]
            nn.BatchNorm2d(feature_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channel * 8, 1, 4, 1, 0, bias=False),                     # [B x 1]
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
    

class DeConvolutionGenerativeAdversarialNetworks(BaseModel):
    def __init__(self, cfg) -> None:
        super(DeConvolutionGenerativeAdversarialNetworks, self).__init__()

        self.cfg          = cfg
        self.device       = cfg.device

        self.image_channel  = cfg.Data.image_shape[0]
        self.latent_size  = cfg.Model.latent_size
        self.gfc = cfg.Model.Generator.feature_channel
        self.dfc = cfg.Model.Discriminator.feature_channel

        self.generator    = Generator(self.latent_size, self.gfc, self.image_channel).to(self.device)
        self.disriminator = Discriminator(self.image_channel, self.dfc).to(self.device)

        self.set_loss()        
        self.set_optimizer()
        self.create_required_directory()

    def set_loss(self):
        self.criterion = nn.BCELoss()

    def set_optimizer(self):
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),    
                                            lr= self.cfg.Optimizer.lr, betas=(self.cfg.Optimizer.beta1, self.cfg.Optimizer.beta2))
        self.d_optimizer = torch.optim.Adam(self.disriminator.parameters(), 
                                            lr= self.cfg.Optimizer.lr, betas=(self.cfg.Optimizer.beta1, self.cfg.Optimizer.beta2))
    
    def create_required_directory(self):
        self.save_image_dir = os.path.join(self.cfg.save_image_path, self.cfg.Model.name, self.cfg.Data.dataset)
        self.save_model_dir = os.path.join(self.cfg.save_model_path, self.cfg.Model.name, self.cfg.Data.dataset)
        os.makedirs(self.save_image_dir, exist_ok=True)
        os.makedirs(self.save_model_dir, exist_ok=True)

    def train(self, data_loader):
        g_losses = []
        d_losses = []
        for epoch in tqdm(range(self.cfg.epochs)):
            for step, (real_img, _) in enumerate(data_loader):
                batch    = real_img.shape[0]
                real_img = real_img.to(self.device)

                # fake image label: 0  real image label: 1 in Discriminator
                # fake image label: 1                      in Generator
                real_label = torch.ones( (batch,), device=self.device)
                fake_label = torch.zeros((batch,), device=self.device)

                # Train Discriminator
                # Gaussian random noise
                noise = torch.randn((batch, self.latent_size, 1, 1), device=self.device)
                gen_img = self.generator(noise)
                real_loss = self.criterion(self.disriminator(real_img).view(-1),         real_label)
                fake_loss = self.criterion(self.disriminator(gen_img.detach()).view(-1), fake_label)
                d_loss    = real_loss + fake_loss

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                noise = torch.randn((batch, self.latent_size, 1, 1), device=self.device)
                gen_img = self.generator(noise)
                g_loss  = self.criterion(self.disriminator(gen_img).view(-1), real_label)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if (step+1) % self.cfg.log_interval_step == 0:
                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())
                    self.print(f'Epoch [{epoch+1}/{self.cfg.epochs}]  Batch [{step+1}/{len(data_loader)}]  Discriminator loss: {d_loss.item():7.4f}  Generator loss: {g_loss.item():7.4f}')

            if (epoch+1) % self.cfg.save_interval_epoch == 0:
                save_image        = self.cfg.denormalize(gen_img)
                save_image_path   = os.path.join(self.save_image_dir, f'epoch_{epoch+1}.png')
                tcvs.utils.save_image(save_image[:25], save_image_path, nrow=5, normalize=True)
                # save_model_path = os.path.join(self.save_model_dir, f'epoch_{epoch+1}.pth')
                # torch.save(self.state_dict(), save_model_path)

        # save loss image
        plt.title(f'Gan training loss on {self.cfg.Data.dataset}')
        plt.plot(g_losses, label='Generator loss')
        plt.plot(d_losses, label='Discriminoator loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_image_dir, 'training_loss.png'))