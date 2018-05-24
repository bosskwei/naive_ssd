import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn import metrics


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        nn.Module.__init__(self)
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.BatchNorm2d(out_channels))
        #
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=padding)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU()
        #
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=padding)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        residual = x
        #
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = self.relu_1(h)
        #
        h = self.conv_2(h)
        h = self.bn_2(h)
        #
        aa = self.reduce(residual)
        h += self.reduce(residual)
        h = self.relu_2(h)
        #
        return h


class ResidualNet(nn.Module):
    def __init__(self, architecture=(2, 2, 2)):
        nn.Module.__init__(self)
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.layer1 = self._make_layer(3, 64, blocks=architecture[0])
        self.layer2 = self._make_layer(64, 128, blocks=architecture[1])
        self.layer3 = self._make_layer(128, 256, blocks=architecture[2])
        self.fc = nn.Sequential(nn.Linear(4096, 1024),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(1024, 10))

    def _make_layer(self, in_channels, out_channels, blocks):
        layer = [BasicBlock(in_channels, out_channels, stride=self.stride, padding=self.padding)]
        for i in range(1, blocks):
            layer.append(BasicBlock(out_channels, out_channels, stride=self.stride, padding=self.padding))
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 4096)
        x = self.fc(x)
        return x


def main():
    train_loader = DataLoader(datasets.CIFAR10(root='./cifar10', train=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010))
                                               ])),
                              batch_size=128, shuffle=True)

    test_loader = DataLoader(datasets.CIFAR10(root='./cifar10', train=False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                              ])),
                             batch_size=200, shuffle=True)

    model = ResidualNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    state = torch.load('./checkpoint/resnet_1_[0-300].pth')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])

    for epoch in range(100):
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            y_predict = model(x_batch)

            loss = criterion(y_predict, target=y_batch)
            loss.backward()
            optimizer.step()

            print('[TRAIN] step: {}, loss: {}'.format(idx, loss))

            if idx % 20 == 0:
                model.eval()
                x_batch, y_batch = next(iter(test_loader))
                y_predict = model(x_batch)
                loss = criterion(y_predict, target=y_batch)
                print('[TEST] loss: {}, accurate: {}'
                      .format(loss, metrics.accuracy_score(y_true=y_batch.detach().numpy(),
                                                           y_pred=np.argmax(y_predict.detach().numpy(), axis=1))))
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           './checkpoint/resnet_1_[{}-{}].pth'.format(epoch, idx))


if __name__ == '__main__':
    main()
