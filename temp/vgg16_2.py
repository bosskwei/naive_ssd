import numpy as np
import torch
from torch import nn
import skimage.data, skimage.transform, skimage.io
import matplotlib.pyplot as plt


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

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            x = layer(x)
            a = torch.max(x)
            b = torch.min(x)
            # draw_layer_output('layer {}: {}'.format(idx, layer), np.squeeze(x.detach().numpy(), axis=0))
        return x


def draw_layer_output(name, tensor):
    print(name)
    c, h, w = tensor.shape

    plt.close()
    fig = plt.figure(figsize=(16, 8))
    for i, img in enumerate(tensor):
        if i >= 64:
            break
        ax = fig.add_subplot(8, 8, i + 1)
        ax.imshow(img)
    plt.pause(5.0)


def main():
    net = VGG16()
    net.load_state_dict(torch.load('models/vgg16.pth'))

    img = np.array(skimage.transform.resize(skimage.io.imread('COCO_train2014_000000381994.jpg'), [512, 512]),
                   dtype=np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = torch.unsqueeze(img, 0)
    net(img)


if __name__ == '__main__':
    main()
