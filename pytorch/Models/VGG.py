import torch
import torch.nn as nn

settings = [64, 64, 'm', 128, 128, 'm', 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512, 'm']

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayers = self._make_conv(settings)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        conv = self.ConvLayers(x)
        flat = conv.view(conv.size(0), -1)
        scores = self.fc(flat)
        return scores
        

    def _make_conv(self, settings):
        layers = []
        in_channels = 3
        for s in settings:
            if s != 'm':
                layers += [nn.Conv2d(in_channels, s, kernel_size=3, padding=1),
                           nn.BatchNorm2d(s),
                           nn.ReLU()]
                in_channels = s
            else:
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        #layers += [nn.AvgPool2d(kernel_size=1,stride=1)]        
        return nn.Sequential(*layers)
