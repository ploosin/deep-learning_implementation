import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, batch_norm=True, activation='') -> None:
        super(ConvLayer, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channel))
        if   activation == 'relu':
            layers.append(nn.ReLU6())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            pass

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    

class DeConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, batch_norm=True, activation='') -> None:
        super(DeConvLayer, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channel))
        if   activation == 'relu':
            layers.append(nn.ReLU6())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            pass

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)