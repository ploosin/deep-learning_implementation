import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import models, utils
import torchvision as tcvs

from Models.basemodel import BaseModel
from Models.conv import ConvLayer, DeConvLayer


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvLayer(3,    64, 4, 2, batch_norm=False, activation='lrelu')        # [B,  64, 128, 128]
        self.conv2 = ConvLayer(64,  128, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 128,  64,  64]
        self.conv3 = ConvLayer(128, 256, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 256,  32,  32]
        self.conv4 = ConvLayer(256, 512, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 512,  16,  16]
        self.conv5 = ConvLayer(512, 512, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 512,   8,   8]
        self.conv6 = ConvLayer(512, 512, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 512,   4,   4]
        self.conv7 = ConvLayer(512, 512, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 512,   2,   2]
        self.conv8 = ConvLayer(512, 512, 4, 2, batch_norm=False, activation='lrelu')        # [B, 512,   1,   1]

        # Decoder
        self.deconv1 = DeConvLayer(512,  512, 4, 2, batch_norm=True,  activation='relu')    # [B, 512,   2,   2]
        self.deconv2 = DeConvLayer(1024, 512, 4, 2, batch_norm=True,  activation='relu')    # [B, 512,   4,   4]
        self.deconv3 = DeConvLayer(1024, 512, 4, 2, batch_norm=True,  activation='relu')    # [B, 512,   8,   8]
        self.deconv4 = DeConvLayer(1024, 512, 4, 2, batch_norm=True,  activation='relu')    # [B, 512,  16,  16]
        self.deconv5 = DeConvLayer(1024, 256, 4, 2, batch_norm=True,  activation='relu')    # [B, 256,  32,  32]
        self.deconv6 = DeConvLayer(512,  128, 4, 2, batch_norm=True,  activation='relu')    # [B, 128,  64,  64]
        self.deconv7 = DeConvLayer(256,  64,  4, 2, batch_norm=True,  activation='relu')    # [B,  64, 128, 128]
        self.deconv8 = DeConvLayer(128,  3,   4, 2, batch_norm=True,  activation='tanh')    # [B,   3, 256, 256]


    def forward(self, x, train=True):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        # Decoder
        d1 = F.dropout(self.deconv1(e8), 0.5, training=train)
        d2 = F.dropout(self.deconv2(torch.cat([d1, e7], dim=1)), 0.5, training=train)
        d3 = F.dropout(self.deconv3(torch.cat([d2, e6], dim=1)), 0.5, training=train)
        d4 = self.deconv4(torch.cat([d3, e5], dim=1))
        d5 = self.deconv5(torch.cat([d4, e4], dim=1))
        d6 = self.deconv6(torch.cat([d5, e3], dim=1))
        d7 = self.deconv7(torch.cat([d6, e2], dim=1))
        y  = self.deconv8(torch.cat([d7, e1], dim=1))
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = ConvLayer(6,    64, 4, 2, batch_norm=False, activation='lrelu')        # [B,  64, 128, 128]
        self.conv2 = ConvLayer(64,  128, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 128,  64,  64]
        self.conv3 = ConvLayer(128, 256, 4, 2, batch_norm=True,  activation='lrelu')        # [B, 256,  32,  32]
        self.conv4 = ConvLayer(256, 512, 4, 1, batch_norm=True,  activation='lrelu')        # [B, 512,  31,  31]
        self.conv5 = ConvLayer(512,   1, 4, 1, batch_norm=True,  activation='')             # [B,   1,  30,  30]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Pix2Pix(BaseModel):
    def __init__(self, cfg) -> None:
        super(Pix2Pix, self).__init__()

        self.cfg          = cfg
        self.device       = cfg.device
        cfg.denormalize = lambda x: ((x+1)/2).clamp(0, 1)
        
        self.generator     = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.set_loss()        
        self.set_optimizer()
        self.create_required_directory()

    def set_loss(self):
        self.L1_loss  = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()

    def set_optimizer(self):
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),     
                                            lr= self.cfg.Optimizer.lr, betas=(self.cfg.Optimizer.beta1, self.cfg.Optimizer.beta2))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                            lr= self.cfg.Optimizer.lr, betas=(self.cfg.Optimizer.beta1, self.cfg.Optimizer.beta2))
    
    def create_required_directory(self):
        self.save_image_dir = os.path.join(self.cfg.save_image_path, self.cfg.Model.name)
        self.save_model_dir = os.path.join(self.cfg.save_model_path, self.cfg.Model.name)
        os.makedirs(self.save_image_dir, exist_ok=True)
        os.makedirs(self.save_model_dir, exist_ok=True)

    def train(self, data_loader):
        for epoch in tqdm(range(self.cfg.epochs)):
            for step, (real_a, real_b) in enumerate(data_loader):
                real_a = real_a.to(self.device)
                real_b = real_b.to(self.device)

                # fake image label: 0  real image label: 1 in Discriminator
                # fake image label: 1                      in Generator
                real_label = torch.ones( 1, device=self.device)
                fake_label = torch.zeros(1, device=self.device)

                # Train Discriminator
                # fake
                fake_b      = self.generator(real_a)                    # [B, 3, 256, 256]
                fake_ab     = torch.cat([real_a, fake_b], dim=1)        # [B, 6, 256, 256]
                pred_fake   = self.discriminator(fake_ab.detach())      # [B, 1,  30,  30]
                loss_d_fake = self.MSE_loss(pred_fake, fake_label)

                # real
                real_ab     = torch.cat([real_a, real_b], dim=1)
                pred_real   = self.discriminator(real_ab)
                loss_d_real = self.MSE_loss(pred_real, real_label)

                # loss
                d_loss = (loss_d_fake + loss_d_real) * 0.5

                self.discriminator.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                fake_ab    = torch.cat([real_a, fake_b], dim=1)
                pred_fake  = self.discriminator(fake_ab)
                loss_g_gan = self.MSE_loss(pred_fake, real_label)

                loss_g_L1  = self.L1_loss(fake_b, real_b) * 10
                
                # loss
                g_loss = loss_g_gan + loss_g_L1

                self.generator.zero_grad()
                self.discriminator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if (step+1) % self.cfg.log_interval_step == 0:
                    self.print(f'Epoch [{epoch+1}/{self.cfg.epochs}]  Batch [{step+1}/{len(data_loader)}]  Discriminator loss: {d_loss.item():7.4f}  Generator loss: {g_loss.item():7.4f}')

            if (epoch+1) % self.cfg.save_interval_epoch == 0:
                save_real_a_image        = self.cfg.denormalize(real_a.squeeze())
                save_real_b_image        = self.cfg.denormalize(real_b.squeeze())
                save_fake_b_image        = self.cfg.denormalize(fake_b.squeeze())
                save_image               = torch.cat([save_real_a_image, save_real_b_image, save_fake_b_image], dim=-1)
                save_image_path   = os.path.join(self.save_image_dir, f'epoch_{epoch+1}.png')
                utils.save_image(save_image, save_image_path)
                # tcvs.utils.save_image(save_image, save_image_path, nrow=5, normalize=True)
                # save_model_path = os.path.join(self.save_model_dir, f'epoch_{epoch+1}.pth')
                # torch.save(self.state_dict(), save_model_path)