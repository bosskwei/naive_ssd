import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn import metrics


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 7
                                      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      # 14
                                      nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.fc = nn.Sequential(nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)
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
                             batch_size=500, shuffle=True)

    #
    # from vgg16_3 import VGG16BN
    # vgg16 = VGG16BN()
    # vgg16.load_state_dict(torch.load('models/vgg16_bn.pth'))

    model = Net()
    # for i in range(24):
    #     model.features[i].load_state_dict(vgg16.features[i].state_dict())

    # model.load_state_dict(torch.load('./checkpoint/temp_2_init.pth'))
    # torch.save(model.state_dict(), './checkpoint/temp_2_init.pth')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    state = torch.load('./checkpoint/temp_2_[0-0].pth')
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
                           './checkpoint/temp_2_[{}-{}].pth'.format(epoch, idx))


if __name__ == '__main__':
    main()
