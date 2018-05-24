import torch
from torch import nn
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 5
                                      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 10
                                      nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 17
                                      nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 24
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, *x):
        return self.features(x)


class VGG16BN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 7
                                      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 14
                                      nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 24
                                      nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 34
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, *x):
        return self.features(x)


def cut_vgg16():
    net1 = torchvision.models.vgg16(pretrained=False)
    states_1 = net1.features.state_dict()

    net2 = VGG16()
    net2.features.load_state_dict(states_1)
    torch.save(net2.state_dict(), 'models/vgg16.pth')


def cut_vgg16_bn():
    net1 = torchvision.models.vgg16_bn(pretrained=False)
    states_1 = net1.features.state_dict()

    net2 = VGG16BN()
    net2.features.load_state_dict(states_1)
    torch.save(net2.state_dict(), 'models/vgg16_bn.pth')


def main():
    cut_vgg16()


if __name__ == '__main__':
    # vgg16 cut fc layers
    main()
