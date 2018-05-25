import numpy as np
import torch
from torch import nn
import skimage.data, skimage.transform, skimage.io
import matplotlib.pyplot as plt


class SSD512(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.feature_layers = nn.ModuleList([nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),
                                             # 6 -> feature_size: 512
                                             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                             nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(128),
                                             nn.ReLU(),
                                             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(128),
                                             nn.ReLU(),
                                             # 13 -> feature_size: 256
                                             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                             nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             # 23 -> feature_size: 128
                                             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                             nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             # 33 -> feature_size: 64
                                             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             # 43 -> feature_size: 32
                                             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)])
        self.extra_layers = nn.ModuleList([nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU(),
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU(),
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU()])
        self.loc_layer = nn.ModuleList([nn.Conv2d(64, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(128, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(256, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
        self.conf_layer = nn.ModuleList([nn.Conv2d(64, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.Conv2d(128, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.Conv2d(256, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.Conv2d(512, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.Conv2d(512, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.Conv2d(512, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])

    def forward(self, x):
        features = []
        loc, conf = [], []

        # extract feature maps
        for i in range(0, 6):
            x = self.feature_layers[i](x)
        features.append(x)

        for i in range(6, 13):
            x = self.feature_layers[i](x)
        features.append(x)

        for i in range(13, 23):
            x = self.feature_layers[i](x)
        features.append(x)

        for i in range(23, 33):
            x = self.feature_layers[i](x)
        features.append(x)

        for i in range(33, 44):
            x = self.feature_layers[i](x)
        features.append(x)

        for i in range(0, 9):
            x = self.extra_layers[i](x)
        features.append(x)

        # loc and conf
        for x, l, c in zip(features, self.loc_layer, self.conf_layer):
            loc.append(l(x).permute(0, 2, 3, 1))
            conf.append(c(x).permute(0, 2, 3, 1))

        return loc, conf


def main():
    net = SSD512()

    img = np.array(skimage.transform.resize(skimage.io.imread('images/COCO_train2014_000000381994.jpg'), [512, 512]),
                   dtype=np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = torch.unsqueeze(img, 0)

    net(img)


if __name__ == '__main__':
    main()
