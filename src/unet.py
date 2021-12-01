"""UNET
"""
import torch
import numpy as np

class UNet(torch.nn.Module):
    """This class implements a UNet for the Segmentation
    """

    def __init__(self, channels):
        """Sets up the U-Net Structure
        """
        super().__init__()
        self.layer1 = self._double_conv(channels, 64)
        self.layer2 = self._double_conv(64, 128)
        self.layer3 = self._double_conv(128, 256)
        self.layer4 = self._double_conv(256, 512)

        #########################################

        self.layer5 = self._double_conv(512 + 256, 256)
        self.layer6 = self._double_conv(256+128, 128)
        self.layer7 = self._double_conv(128+64, 64)
        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)

        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        #x5 = torch.nn.ConvTranspose2d(512, 512, 2, 2)(x4)
        x5 = torch.cat([x5, x3], dim=1)       
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)        
        #x6 = torch.nn.ConvTranspose2d(256, 256, 2, 2)(x5)
        x6 = torch.cat([x6, x2], dim=1)       
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        #x7 = torch.nn.ConvTranspose2d(128, 128, 2, 2)(x6)
        x7 = torch.cat([x7, x1], dim=1)       
        x7 = self.layer7(x7)

        ret = self.layer8(x7)
        return ret


    def _double_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(inplace=True)
    )   
