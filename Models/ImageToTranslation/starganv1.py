import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import models, utils
from torchsummary import summary
import torchvision as tcvs
from matplotlib import pyplot as plt

from Models.basemodel import BaseModel
from Models.conv import ConvLayer, DeConvLayer, ResidualBlock
from Loss.loss import gradient_penalty

class Generator(nn.Module):
    def __init__(self, input_channel=5, hidden_channel=64):
        super(Generator, self).__init__()

        # DownSampling
        self.conv1 = ConvLayer(input_channel+3,  hidden_channel*1, 7, 1, 3, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.conv2 = ConvLayer(hidden_channel*1, hidden_channel*2, 4, 2, 1, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.conv3 = ConvLayer(hidden_channel*2, hidden_channel*4, 4, 2, 1, batch_norm=False, instance_norm=True, activation='relu', bias=False)

        # BottleNeck
        self.res1 = ResidualBlock(hidden_channel*4, hidden_channel*4, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.res2 = ResidualBlock(hidden_channel*4, hidden_channel*4, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.res3 = ResidualBlock(hidden_channel*4, hidden_channel*4, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.res4 = ResidualBlock(hidden_channel*4, hidden_channel*4, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.res5 = ResidualBlock(hidden_channel*4, hidden_channel*4, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.res6 = ResidualBlock(hidden_channel*4, hidden_channel*4, batch_norm=False, instance_norm=True, activation='relu', bias=False)

        # UpSampling
        self.deconv1 = DeConvLayer(hidden_channel*4,  hidden_channel*2, 4, 2, 1, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.deconv2 = DeConvLayer(hidden_channel*2,  hidden_channel*1, 4, 2, 1, batch_norm=False, instance_norm=True, activation='relu', bias=False)
        self.conv4   = ConvLayer(hidden_channel, 3, 7, 1, 3, batch_norm=False, instance_norm=False, activation='', bias=False)
        self.tanh    = nn.Tanh()

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)      # [B, 5,      1,     1]
        c = c.repeat(1, 1, x.size(2), x.size(3))    # [B, 5, height, width]
        x = torch.cat([x, c], dim=1)

        # DownSampling
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # BottleNeck
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        # UpSampling
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_size=128, input_channel=5, hidden_channel=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = ConvLayer(                3,  hidden_channel* 1, 4, 2, 1, batch_norm=False, activation='lrelu', bias=False)    # [B,   64, 64, 64]
        self.conv2 = ConvLayer(hidden_channel* 1,  hidden_channel* 2, 4, 2, 1, batch_norm=False, activation='lrelu', bias=False)    # [B,  128, 32, 32]
        self.conv3 = ConvLayer(hidden_channel* 2,  hidden_channel* 4, 4, 2, 1, batch_norm=False, activation='lrelu', bias=False)    # [B,  256, 16, 16]
        self.conv4 = ConvLayer(hidden_channel* 4,  hidden_channel* 8, 4, 2, 1, batch_norm=False, activation='lrelu', bias=False)    # [B,  512,  8,  8]
        self.conv5 = ConvLayer(hidden_channel* 8,  hidden_channel*16, 4, 2, 1, batch_norm=False, activation='lrelu', bias=False)    # [B, 1024,  4,  4]
        self.conv6 = ConvLayer(hidden_channel*16,  hidden_channel*32, 4, 2, 1, batch_norm=False, activation='lrelu', bias=False)    # [B, 2048,  2,  2]

        # kernel_size = int(img_size / np.power(2, 6))
        self.gen = ConvLayer(hidden_channel*32,             1, 4, 1, 1, batch_norm=False, bias=False)   # [B, 1,  1,  1]
        self.cls = ConvLayer(hidden_channel*32, input_channel, 4, 2, 1, batch_norm=False, bias=False)   # [B, 5,  1,  1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        gen = self.gen(x)   # [B, 1]
        cls = self.cls(x)   # [B, 5]
        return gen, cls.view(cls.size(0), cls.size(1))

class StarGANv1(BaseModel):
    def __init__(self, cfg) -> None:
        super(StarGANv1, self).__init__()

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

    def create_labels(self, label, input_channels=5, selected_attrs=None):
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        label_target_list = []
        for i in range(input_channels):
            label_target = label.clone()
            if i in hair_color_indices:
                label_target[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        label_target[:, j] = 0
            else:
                label_target[:, i] = (label_target[:, i] == 0)
            
            label_target_list.append(label_target.to(self.device))
        return label_target_list

    def train(self, data_loader):

        celeba_iter  = iter(data_loader)
        fixed_image, fixed_label = next(celeba_iter)
        fixed_image = fixed_image.to(self.device)

        for step in tqdm(range(self.cfg.iteration)):
            try:
                real_x, label_org = next(celeba_iter)
            except:
                celeba_iter = iter(data_loader)
                real_x, label_org = next(celeba_iter)
            
            batch = real_x.size(0)
            rand_idx = torch.randperm(batch)
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            real_x = real_x.to(self.device)
            c_org  =  c_org.to(self.device)
            c_trg  =  c_trg.to(self.device)
            label_org = label_org.to(self.device)
            label_trg = label_trg.to(self.device)

            # Train Discriminator
            ## compute loss with real images
            src, cls    = self.discriminator(real_x)
            loss_d_real = -torch.mean(src)
            loss_d_cls  = F.binary_cross_entropy_with_logits(cls, label_org)

            ## compute loss with fake images
            fake_x      = self.generator(real_x, c_trg)
            src, cls    = self.discriminator(fake_x.detach())
            loss_d_fake = torch.mean(src)

            ## compute loss for gradient penalty
            alpha     = torch.rand(batch, 1, 1, 1).to(self.device)
            x_hat     = (alpha * real_x.data + (1-alpha) * fake_x.data).requires_grad_(True)
            src, _    = self.discriminator(x_hat)
            loss_d_gp = gradient_penalty(src, x_hat, device=self.device)

            ## backward and optimize
            loss_d = loss_d_real + loss_d_fake + 1.0 * loss_d_cls + 10.0 * loss_d_gp
            self.discriminator.zero_grad()
            loss_d.backward()
            self.d_optimizer.step()

            # logging
            losses = {}
            losses['D/real'] = loss_d_real.item()
            losses['D/fake'] = loss_d_fake.item()
            losses['D/cls']  = loss_d_cls.item()
            losses['D/gp']   = loss_d_gp.item()

            # Train Generator
            if (step + 1) % 5 == 0:
                # original to target domain
                fake_x      = self.generator(real_x, c_trg)
                src, cls    = self.discriminator(fake_x) 
                loss_g_fake = -torch.mean(src)
                loss_g_cls  = F.binary_cross_entropy_with_logits(cls, label_trg)

                # target to original domain
                reconst_x  = self.generator(fake_x, c_org)
                loss_g_rec = torch.mean(torch.abs(real_x - reconst_x))

                # backward and optimize
                loss_g = loss_g_fake + 10.0 * loss_g_rec + 1.0 * loss_g_cls
                self.generator.zero_grad()
                loss_g.backward()
                self.g_optimizer.step()

                # logging
                losses['G/fake'] = loss_g_fake.item()
                losses['G/rec']  = loss_g_rec.item()
                losses['G/cls']  = loss_g_cls.item()

            if (step+1) % self.cfg.log_interval_step == 0:
                self.print(step+1, self.cfg.iteration, losses)

                # visualize
                with torch.no_grad():
                    fixed_label_list = self.create_labels(fixed_label, 5, selected_attrs=self.cfg.Data.selected_attrs)

                    ## Translate images
                    fake_x_list = [fixed_image]
                    for fl in fixed_label_list:
                        fake_x_list.append(self.generator(fixed_image, fl))

                    ## Save image
                    img_path = os.path.join(self.save_image_dir, f'{step+1}.png')
                    img_conc = torch.cat(fake_x_list, dim=3)
                    grid = tcvs.utils.make_grid(self.cfg.denormalize(img_conc.data.cpu()), nrow=1, padding=0)
                    
                    plt.figure(figsize=(10, 20))
                    plt.imshow(grid.permute(1, 2, 0))
                    plt.savefig(img_path)


            if (step+1) % self.cfg.save_interval_step == 0:
                g_path = os.path.join(self.save_model_dir, f'{step+1}-G.pth')
                d_path = os.path.join(self.save_model_dir, f'{step+1}-D.pth')
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)