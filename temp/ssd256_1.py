import numpy as np
import xml.etree.ElementTree as etree
import torch
from torch import nn
from torch.utils import data
from skimage import io, transform
from ssd.loader_2 import generate_prior_boxes, match


def image_padding(image, boxes, target_shape=(512, 512)):
    h, w, c = np.shape(image)
    target_h, target_w = target_shape

    # compute scale
    scale = target_w / w if h < w else target_h / h
    image = transform.rescale(image, scale, mode='reflect')

    # compute offset
    h, w, c = np.shape(image)
    offset_h, offset_w = (target_h - h) // 2, (target_w - w) // 2

    # fill into target
    target_image = np.zeros(shape=[target_h, target_w, c], dtype=np.float32)
    target_image[offset_h:offset_h + h, offset_w:offset_w + w, :] = image

    # adaptive boxes
    boxes_new = []
    for top, bottom, left, right, name in boxes:
        left, right = left * scale + offset_w, right * scale + offset_w
        top, bottom = top * scale + offset_h, bottom * scale + offset_h
        boxes_new.append([top, bottom, left, right, name])

    #
    return target_image, boxes_new


class VOC2007(data.Dataset):
    ROOT = '/home/bosskwei/data/VOCdevkit/VOC2007'
    IMAGE_FOLDER = 'JPEGImages'
    LABEL_FOLDER = 'Annotations'
    START_IDX, END_IDX = 1, 9963

    label_index = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                   'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
                   'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                   'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

    def __init__(self, mode='train'):
        self._mode = mode
        self._prior_boxes = generate_prior_boxes()

    def __len__(self):
        return self.END_IDX - self.START_IDX + 1

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError
        if item > len(self):
            raise IndexError

        # read image
        image = io.imread('{0}/{1}/{2:06d}.jpg'.format(self.ROOT, self.IMAGE_FOLDER, item))

        # read label
        boxes = []
        tree = etree.parse('{0}/{1}/{2:06d}.xml'.format(self.ROOT, self.LABEL_FOLDER, item))
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            left, right = int(box.find('xmin').text), int(box.find('xmax').text)
            top, bottom = int(box.find('ymin').text), int(box.find('ymax').text)
            boxes.append([top, bottom, left, right, name])

        # padding images to same size
        image, boxes = image_padding(image, boxes, target_shape=(256, 256))

        # if not training, just return images and raw boxes
        if self._mode != 'train':
            return image, boxes

        # permute from (h, w, c) to (c, h, w)
        image = np.swapaxes(image, 2, 0)

        # match prior boxes and construct
        loc, label = np.zeros(shape=(1040, 4), dtype=np.float32), np.zeros(shape=1040, dtype=np.int)
        matched_idx = match(prior_boxes=self._prior_boxes, boxes=boxes)
        # print('len_boxes:', len(boxes), 'matched:', matched_idx)
        assert len(matched_idx) == len(boxes)
        for mi, box in zip(matched_idx, boxes):
            top, bottom, left, right, name = box

            loc[mi, :] = top, bottom, left, right
            label[mi] = self.label_index[name]

            # TODO: 用偏移坐标取代绝对坐标

        return image, loc, label


class SSD256(nn.Module):
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
                                           nn.ReLU(),
                                           nn.MaxPool2d(kernel_size=2, stride=2, padding=0)])
        self.loc_layer = nn.ModuleList([nn.Conv2d(512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
        self.conf_layer = nn.ModuleList([nn.Conv2d(512, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.Conv2d(512, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])

    def forward(self, x):
        """
        :param x: input image with shape = (N, C, H, W)
        :return:
        """
        features = []
        loc, conf = [], []

        # extract feature maps
        # (to simplify the process, we only extract 3 feature maps)
        for i in range(0, 33):
            x = self.feature_layers[i](x)
        features.append(x)

        for i in range(33, 44):
            x = self.feature_layers[i](x)

        for i in range(0, 10):
            x = self.extra_layers[i](x)
        features.append(x)

        # loc and conf
        assert len(features) == len(self.loc_layer) == len(self.conf_layer)
        for x, l, c in zip(features, self.loc_layer, self.conf_layer):
            loc.append(l(x).permute(0, 2, 3, 1))
            conf.append(c(x).permute(0, 2, 3, 1))

        # current shape:
        # loc: [(n, 32, 32, 4), (n, 4, 4, 4)] -> (small, big) objects
        # conf: [(n, 32, 32, 21), (n, 4, 4, 21)] -> (small, big) objects

        return loc, conf


class MultiBoxLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.loc_loss_fn = nn.L1Loss()
        self.conf_loss_fn = nn.CrossEntropyLoss(reduce=False)

    def forward(self, loc_predict, conf_predict, loc_target, label_target):
        # current shape:
        # loc_predict: [(n, 32, 32, 4), (n, 4, 4, 4)] -> (small, big) objects
        # conf_predict: [(n, 32, 32, 21), (n, 4, 4, 21)] -> (small, big) objects

        # reshape array and concrete them
        n, _, _, _ = loc_predict[0].shape
        loc_predict = [item.view(n, -1, 4) for item in loc_predict]
        conf_predict = [item.view(n, -1, 21) for item in conf_predict]
        loc_predict, conf_predict = torch.cat(loc_predict, dim=1), torch.cat(conf_predict, dim=1)

        # current shape:
        # loc_predict / loc_target: (n, 1040, 4), 1040 = 32 * 32 + 4 * 4
        # conf_predict / conf_target: (n, 1040, 21)
        assert loc_predict.shape == loc_target.shape
        assert conf_predict.shape == (n, label_target.shape[1], 21)

        #
        loc_predict, loc_target = loc_predict.view(-1, 4), loc_target.view(-1, 4)
        conf_predict, label_target = conf_predict.view(-1, 21), label_target.view(-1)

        #
        positive_idx = label_target != 0
        num_positive = positive_idx.sum()
        num_negative = 3 * num_positive

        # loc loss
        loss_loc = self.loc_loss_fn(input=loc_predict[positive_idx], target=loc_target[positive_idx])

        # conf loss with hard negative mining
        loss_conf_temp = self.conf_loss_fn(input=conf_predict, target=label_target)
        #
        loss_conf_neg = loss_conf_temp[1.0 - positive_idx]
        _, indices = torch.sort(loss_conf_neg, descending=True)
        negative_idx = indices[:num_negative]
        loss_conf = torch.cat((loss_conf_temp[positive_idx], loss_conf_temp[negative_idx]))
        loss_conf = torch.mean(loss_conf)

        #
        return loss_loc + loss_conf


def main():
    net = SSD256()
    criterion = MultiBoxLoss()

    loader = data.DataLoader(dataset=VOC2007(), batch_size=2, shuffle=True)

    for image_batch, loc_batch, label_batch in loader:
        loc_predict, conf_predict = net(image_batch)
        loss = criterion(loc_predict=loc_predict,
                         conf_predict=conf_predict,
                         loc_target=loc_batch,
                         label_target=label_batch)
        a = 0


if __name__ == '__main__':
    main()
