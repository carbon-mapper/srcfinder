import torch

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3),
                            padding='valid'),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3),
                            padding='valid'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PadDoubleConv(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(PadDoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3),
                            padding='same', padding_mode='replicate'),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3),
                            padding='same', padding_mode='replicate'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class StandardUp(torch.nn.Module):
    def __init__(self, in_channels, out_channels, tanh=False):
        super(StandardUp, self).__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Crop x2 to match x1
        crop4 = x1.size(dim=3) - x2.size(dim=3)
        crop3 = x1.size(dim=2) - x2.size(dim=2)
        x2 = torch.nn.functional.pad(x2, [
            crop4 // 2, crop4 - (crop4 // 2),
            crop3 // 2, crop3 - (crop3 // 2)
        ])

        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)

        return x

class PadStandardUp(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PadStandardUp, self).__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
        self.conv = PadDoubleConv(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Crop x2 to match x1
        crop4 = x1.size(dim=3) - x2.size(dim=3)
        crop3 = x1.size(dim=2) - x2.size(dim=2)
        x2 = torch.nn.functional.pad(x2, [
            crop4 // 2, crop4 - (crop4 // 2),
            crop3 // 2, crop3 - (crop3 // 2)
        ])
        
        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)

        return x

class OneByOneConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OneByOneConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)

        return x

class PaddedUNet(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=1, init_weights=True):
        super(PaddedUNet, self).__init__()
        self.in_ch = in_ch
        self.pool = torch.nn.MaxPool2d(2)
        self.input = PadDoubleConv(self.in_ch, 64, 64)
        self.down1 = PadDoubleConv(64, 128, 128)
        self.down2 = PadDoubleConv(128, 256, 256)
        self.down3 = PadDoubleConv(256, 512, 512)
        self.down4 = PadDoubleConv(512, 1024, 1024)
        self.up1 = PadStandardUp(1024, 512)
        self.up2 = PadStandardUp(512, 256)
        self.up3 = PadStandardUp(256, 128)
        self.up4 = PadStandardUp(128, 64)
        self.output = OneByOneConv(64, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or \
               isinstance(m, torch.nn.ConvTranspose2d) or \
               isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input to feature
        x1 = self.input(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        xo = self.output(xo)

        return xo

class DeepPaddedUNet(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=1, init_weights=True):
        super(DeepPaddedUNet, self).__init__()
        self.in_ch = in_ch
        self.pool = torch.nn.MaxPool2d(2)
        self.input = PadDoubleConv(self.in_ch, 32, 32)
        self.down1 = PadDoubleConv(32, 64, 64)
        self.down2 = PadDoubleConv(64, 128, 128)
        self.down3 = PadDoubleConv(128, 256, 256)
        self.down4 = PadDoubleConv(256, 512, 512)
        self.down5 = PadDoubleConv(512, 1024, 1024)
        
        self.up1 = PadStandardUp(1024, 512)
        self.up2 = PadStandardUp(512, 256)
        self.up3 = PadStandardUp(256, 128)
        self.up4 = PadStandardUp(128, 64)
        self.up5 = PadStandardUp(64, 32)
        self.output = OneByOneConv(32, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or \
               isinstance(m, torch.nn.ConvTranspose2d) or \
               isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)        

    def forward(self, x):
        # Input to feature
        x1 = self.input(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        x6 = self.down5(self.pool(x5))

        xo = self.up1(x6, x5)
        xo = self.up2(xo, x4)
        xo = self.up3(xo, x3)
        xo = self.up4(xo, x2)
        xo = self.up5(xo, x1)
        xo = self.output(xo)

        return xo    
    
class UNet(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=1, init_weights=True):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.pool = torch.nn.MaxPool2d(2)
        self.input = DoubleConv(self.in_ch, 64, 64)
        self.down1 = DoubleConv(64, 128, 128)
        self.down2 = DoubleConv(128, 256, 256)
        self.down3 = DoubleConv(256, 512, 512)
        self.down4 = DoubleConv(512, 1024, 1024)
        self.up1 = StandardUp(1024, 512)
        self.up2 = StandardUp(512, 256)
        self.up3 = StandardUp(256, 128)
        self.up4 = StandardUp(128, 64)
        self.output = OneByOneConv(64, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or \
               isinstance(m, torch.nn.ConvTranspose2d) or \
               isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)        


    def forward(self, x):
        # Input to feature
        x1 = self.input(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        xo = self.output(xo)

        return xo

class UNetLite(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=1, init_weights=True):
        super(UNetLite, self).__init__()
        self.in_ch = in_ch
        self.pool = torch.nn.MaxPool2d(2)
        self.input = PadDoubleConv(self.in_ch, 16, 16)
        self.down1 = PadDoubleConv(16, 32, 32)
        self.down2 = PadDoubleConv(32, 64, 64)
        self.down3 = PadDoubleConv(64, 128, 128)
        self.down4 = PadDoubleConv(128, 256, 256)

        self.up1 = PadStandardUp(256, 128)
        self.up2 = PadStandardUp(128, 64)
        self.up3 = PadStandardUp(64, 32)
        self.up4 = PadStandardUp(32, 16)
        self.output = OneByOneConv(16, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or \
               isinstance(m, torch.nn.ConvTranspose2d) or \
               isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)        


    def forward(self, x):
        # Input to feature
        x1 = self.input(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        xo = self.output(xo)

        return xo


class CUNetLite(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=1, init_weights=True):
        super(CUNetLite, self).__init__()
        self.in_ch = in_ch
        self.pool = torch.nn.MaxPool2d(2)
        self.input = PadDoubleConv(self.in_ch, 16, 16)
        self.down1 = PadDoubleConv(16, 32, 32)
        self.down2 = PadDoubleConv(32, 64, 64)
        self.down3 = PadDoubleConv(64, 128, 128)
        self.down4 = PadDoubleConv(128, 256, 256)
        
        self.outputc = OneByOneConv(256, num_classes)
        self.maxpool = torch.nn.AdaptiveMaxPool2d((1,1))
    
        self.up1 = PadStandardUp(256, 128)
        self.up2 = PadStandardUp(128, 64)
        self.up3 = PadStandardUp(64, 32)
        self.up4 = PadStandardUp(32, 16)
        self.output = OneByOneConv(16, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or \
               isinstance(m, torch.nn.ConvTranspose2d) or \
               isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)        

    def forward(self, x):
        # Input to feature
        x1 = self.input(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        xc = self.outputc(x5)
        xc = self.maxpool(xc)
        xc = torch.flatten(xc, 1)
        
        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        xo = self.output(xo)

        return xo,xc