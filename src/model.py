import torch
import torch.nn as nn
from torch import Tensor
from func import *
import torchvision.models as models
import torchvision.models.quantization as models_q


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return tanh(sigmoid, torch.exp, input)


class Conv2dActivion(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode: str = 'zeros',
            activation: nn.Module = nn.ReLU):
        super(Conv2dActivion, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode
        )
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x


class Conv2dBatchNormActivion(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode: str = 'zeros',
            activation: nn.Module = nn.ReLU):
        super(Conv2dBatchNormActivion, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LinearActivion(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: nn.Module = nn.ReLU):
        super(LinearActivion, self).__init__()
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias,
        )
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.act(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, n_classes, activation: nn.Module):
        super(LeNet5, self).__init__()
        features = [6, 16, 120, 84]
        self.feature_extractor = nn.Sequential(
            Conv2dActivion(in_channels=1, out_channels=features[0],
                           kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2),
            Conv2dActivion(in_channels=features[0], out_channels=features[1],
                           kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2),
            Conv2dActivion(in_channels=features[1], out_channels=features[2],
                           kernel_size=4, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=features[2], out_features=features[3]),
            activation(),
            nn.Linear(in_features=features[3], out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class LeNet5_ReLU(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5_ReLU, self).__init__()
        self.net = LeNet5(n_classes, nn.ReLU)

    def forward(self, x):
        return self.net(x)


class LeNet5_Tanh(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5_Tanh, self).__init__()
        self.net = LeNet5(n_classes, nn.Tanh)

    def forward(self, x):
        return self.net(x)


class SmallNet(nn.Module):
    def __init__(self, n_classes):
        super(SmallNet, self).__init__()
        features = [64, 64, 128, 128, 256]
        self.feature_extractor = nn.Sequential(
            Conv2dBatchNormActivion(
                in_channels=3, out_channels=features[0], kernel_size=3, stride=2),
            Conv2dBatchNormActivion(
                in_channels=features[0], out_channels=features[1], kernel_size=3, stride=1),
            Conv2dBatchNormActivion(
                in_channels=features[1], out_channels=features[2], kernel_size=3, stride=2),
            Conv2dBatchNormActivion(
                in_channels=features[2], out_channels=features[3], kernel_size=3, stride=1),
            Conv2dBatchNormActivion(
                in_channels=features[3], out_channels=features[4], kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=features[4], out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class QResNet18(models_q.QuantizableResNet):
    def __init__(self, n_classes):
        super(QResNet18, self).__init__(
            models_q.resnet.QuantizableBasicBlock, [2, 2, 2, 2])
        if n_classes != 1000:
            self.fc = nn.Linear(self.fc.in_features, n_classes)
        self.quant = nn.Identity()
        self.dequant = nn.Identity()

    def forward(self, x):
        return self._forward_impl(x)


class QResNet18_32x32(QResNet18):
    def __init__(self, n_classes):
        super(QResNet18_32x32, self).__init__(
            100)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))


class VGG16(models.VGG):
    def __init__(
            self,
            num_classes=1000,
            init_weights=False):
        super(VGG16, self).__init__(
            models.vgg.make_layers(models.vgg.cfgs['D'], batch_norm=True), num_classes, init_weights)


class VGG16_32x32(VGG16):
    def __init__(
            self,
            num_classes=100,
            init_weights=False):
        super(VGG16_32x32, self).__init__(num_classes, init_weights)
        self.avgpool = nn.Identity()
        self.classifier[0] = nn.Linear(in_features=512, out_features=4096)

class VGG11(models.VGG):
    def __init__(
            self,
            num_classes=1000,
            init_weights=False):
        super(VGG11, self).__init__(
            models.vgg.make_layers(models.vgg.cfgs['A'], batch_norm=True), num_classes, init_weights)


class VGG11_32x32(VGG11):
    def __init__(
            self,
            num_classes=100,
            init_weights=False):
        super(VGG11_32x32, self).__init__(num_classes, init_weights)
        self.avgpool = nn.Identity()
        self.classifier[0] = nn.Linear(in_features=512, out_features=4096)
        for i in [3,4,5]:
            self.classifier[i] = nn.Identity()

if __name__ == '__main__':
    from data_loader import data_loader
    # train_loader, valid_loader = data_loader('CIFAR10', 1)

    # small_net = SmallNet(10)
    # for X, y_true in train_loader:
    #     out = small_net(X)
    #     print(out.shape)
    #     break
