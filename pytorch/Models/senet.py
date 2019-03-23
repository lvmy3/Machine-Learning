import torch
import torch.nn as nn
import torchvision.models as models

class SEblock(nn.Module):
    def __init__(self, inplanes, r=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential([
            nn.Linear(inplanes, inplanes // r, bias=False)
            nn.ReLU(inplace=True)
            nn.Linear(inplanes // r, inplanes, bias=False)
            nn.Sigmoid()
        ])
    
    def forward(self, x):
        num, channels, _, _ = x.size()
        output = self.avgpool(x).view(num, channels)
        output = self.fc(output).view(num, channels, 1, 1)
        return tmp * output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#SE verion of resnet basic block
class SE_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, r=16, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.seblock = SEblock(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.seblock(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SE_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, r=16, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.seblock = SEblock(planes * self.expansion, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.seblock(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def se_resnet18(num_classes):
    model = models.ResNet(SE_BasicBlock, [2, 2, 2, 2], num_classes)
    return model

def se_resnet50(num_classes):
    model = models.ResNet(SE_Bottleneck, [3, 4, 6, 3], num_classes)