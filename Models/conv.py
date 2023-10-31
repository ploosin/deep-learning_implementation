import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, batch_norm=True, instance_norm=False, activation='', bias=True) -> None:
        super(ConvLayer, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, bias=bias))
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channel))
        elif instance_norm:
            layers.append(nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True))

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
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, batch_norm=True, instance_norm=False, activation='', bias=True) -> None:
        super(DeConvLayer, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding, bias=bias))
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channel))
        elif instance_norm:
            layers.append(nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True))
        
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
    

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, batch_norm=True, instance_norm=False, activation='', bias=True) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(input_channel,  output_channel, kernel_size, stride, padding, batch_norm, instance_norm, activation, bias=bias)
        self.conv2 = ConvLayer(output_channel, output_channel, kernel_size, stride, padding, batch_norm, instance_norm, bias=bias)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y