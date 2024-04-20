import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.cfg = cfg
        self.conv_in = nn.Conv2d(3, cfg.in_channels, kernel_size=3, stride=1, padding=1)
        self.resnets = nn.ModuleList([])
        for i in range(cfg.n_resnet_blocks):
            ch_in = cfg.in_channels * (2 ** i)
            ch_out = cfg.in_channels * (2 ** (i + 1))
            self.resnets.append(ResnetBlock(ch_in, ch_out, cfg.norm_groups, cfg.dropout_prob))
            if i < 2:
                self.resnets.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.fl = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * ch_out, 512)
        self.dropout = nn.Dropout(cfg.dropout_prob)
        self.out = nn.Linear(512, 10)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.conv_in(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.fl(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

class ResnetBlock(nn.Module):
    """
    Residual block with group normalization, SiLU activation, and dropout.

    Args:
        - in_channels: number of input channels
        - out_channels: number of output channels
        - norm_groups: number of groups for group normalization
        - dropout_prob: dropout probability

    Inputs:
        - x: input tensor of shape (B, C, H, W)

    Outputs:
        - output tensor of shape (B, out_channels, H, W)
    """
    def __init__(self,
            in_channels: int,
            out_channels: int,
            norm_groups: int,
            dropout_prob: float,
        ):
        super(ResnetBlock, self).__init__()
        self.net1 = nn.Sequential(
            nn.GroupNorm(norm_groups, in_channels),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        if hasattr(self, 'skip_conv'):
            x = self.skip_conv(x)
        return x + out

        
