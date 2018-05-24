import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset
from skimage import io
import xml.etree.ElementTree as etree


class VOC2007(Dataset):
    ROOT = '/home/bosskwei/data/VOCdevkit/VOC2007'
    IMAGE = 'JPEGImages'
    LABEL = 'Annotations'
    START_IDX, END_IDX = 1, 9963

    def __init__(self):
        pass

    def __len__(self):
        return self.END_IDX - self.START_IDX + 1

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError
        if item > len(self):
            raise IndexError

        # read image
        image = io.imread('{0}/{1}/{2:06d}.jpg'.format(self.ROOT, self.IMAGE, item))

        # read label
        objects = []
        tree = etree.parse('{0}/{1}/{2:06d}.xml'.format(self.ROOT, self.LABEL, item))
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            left, right = int(box.find('xmin').text), int(box.find('xmax').text)
            top, bottom = int(box.find('ymin').text), int(box.find('ymax').text)
            objects.append([name, left, right, top, bottom])

        return image, objects


def test_voc2007():
    voc2007 = VOC2007()
    #
    fig, ax = plt.subplots()
    image, objects = voc2007[random.randrange(len(voc2007))]
    plt.imshow(image)
    for name, left, right, top, bottom in objects:
        color = random.choice(['red', 'green', 'blue'])
        plt.text(left, top, name, color=color)
        rect = patches.Rectangle((left, top), (right - left), (bottom - top), edgecolor=color, linewidth=2, fill=False)
        ax.add_patch(rect)
    #
    fig, ax = plt.subplots()
    image, objects = voc2007[random.randrange(len(voc2007))]
    plt.imshow(image)
    for name, left, right, top, bottom in objects:
        color = random.choice(['red', 'green', 'blue'])
        plt.text(left, top, name, color=color)
        rect = patches.Rectangle((left, top), (right - left), (bottom - top), edgecolor=color, linewidth=2, fill=False)
        ax.add_patch(rect)
    plt.show()


def main():
    test_voc2007()


def prior_boxes():
    pass


if __name__ == '__main__':
    main()
