import os
import torch
import torch.nn as nn
import torchvision as tcvs
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from Models.basemodel import BaseModel

class Generator(nn.Module):
    def __init__(self, image_shape, input_channel, output_channel) -> None:
        super(Generator, self).__init__()
        
        self.image_shape = image_shape
        self.model = nn.Sequential(
            *self.block(input_channel, 128, batch_norm=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, output_channel),
            nn.Tanh()
        )

    def block(self, input_channel, output_channel, batch_norm=True):
        layers = [nn.Linear(input_channel, output_channel)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_channel))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, x):
        batch = x.shape[0]
        x = self.model(x)
        return x.reshape(batch , *self.image_shape)

class Discriminator(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_channel, output_channel*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(output_channel*2, output_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(output_channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        batch = img.shape[0]
        x = img.reshape(batch, -1)
        return self.model(x)
    

class GenerativeAdversarialNetworks(BaseModel):
    def __init__(self, cfg) -> None:
        super(GenerativeAdversarialNetworks, self).__init__()

        self.cfg          = cfg
        self.device       = cfg.device
        self.image_shape  = cfg.Data.image_shape

        self.image_flatten_channel  = int(np.prod(self.image_shape))
        self.output_dis_channel     = cfg.Model.Discriminator.output_channel
        self.latent_size            = cfg.Model.latent_size

        self.generator    = Generator(self.image_shape, self.latent_size, self.image_flatten_channel).to(self.device)
        self.disriminator = Discriminator(self.image_flatten_channel, self.output_dis_channel).to(self.device)

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
                real_label = torch.ones( (batch, 1), device=self.device)
                fake_label = torch.zeros((batch, 1), device=self.device)

                # Train Discriminator
                # Gaussian random noise
                noise = torch.randn((batch, self.latent_size), device=self.device)
                gen_img = self.generator(noise)
                real_loss = self.criterion(self.disriminator(real_img),         real_label)
                fake_loss = self.criterion(self.disriminator(gen_img.detach()), fake_label)
                d_loss    = (real_loss + fake_loss) * 0.5

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                noise   = torch.randn((batch, self.latent_size), device=self.device)
                gen_img = self.generator(noise)
                g_loss  = self.criterion(self.disriminator(gen_img), real_label)

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
