import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset
from skimage import io, transform
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

        # return
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
    plt.show()


class VOC2012(Dataset):
    ROOT = '/home/bosskwei/data/VOCdevkit/VOC2012'
    IMAGE = 'JPEGImages'
    LABEL = 'Annotations'

    def __init__(self):
        # find all annotations
        self.labels = []
        for child in pathlib.Path('{0}/{1}'.format(self.ROOT, self.LABEL)).iterdir():
            if not child.is_file():
                continue
            filename, objects = self._parse_label(child)
            self.labels.append([filename, objects])

    def is_not_used(self):
        pass

    def _parse_label(self, file):
        self.is_not_used()
        # read label
        objects = []
        tree = etree.parse(file)
        filename = tree.getroot().find('filename').text
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            left, right = float(box.find('xmin').text), float(box.find('xmax').text)
            top, bottom = float(box.find('ymin').text), float(box.find('ymax').text)
            objects.append([name, left, right, top, bottom])
        return filename, objects

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError
        if item > len(self):
            raise IndexError

        # read label
        filename, label = self.labels[item]

        # read image
        image = io.imread('{0}/{1}/{2}'.format(self.ROOT, self.IMAGE, filename))

        # return
        return image, label


def test_voc2012():
    voc2012 = VOC2012()
    #
    while True:
        fig, ax = plt.subplots()
        image, objects = voc2012[random.randrange(len(voc2012))]
        plt.imshow(image)
        for name, left, right, top, bottom in objects:
            color = random.choice(['red', 'green', 'blue'])
            plt.text(left, top, name, color=color)
            rect = patches.Rectangle((left, top), (right - left), (bottom - top),
                                     edgecolor=color, linewidth=2, fill=False)
            ax.add_patch(rect)
        plt.pause(2.0)
        plt.close()


def image_padding(image, target_shape=(512, 512)):
    h, w, c = np.shape(image)
    target_h, target_w = target_shape

    # compute scale
    scale = target_w / w if h < w else target_h / h
    image = transform.rescale(image, scale, mode='reflect')

    # compute offset
    h, w, c = np.shape(image)
    offset_h, offset_w = (target_h - h) // 2, (target_w - w) // 2

    # fill into target
    target_image = np.zeros(shape=[target_h, target_w, c], dtype=image.dtype)
    target_image[offset_h:offset_h + h, offset_w:offset_w + w, :] = image

    #
    return target_image, scale, offset_w, offset_h


def test_image_padding():
    voc2012 = VOC2012()
    while True:
        fig, ax = plt.subplots()
        image, objects = voc2012[random.randrange(len(voc2012))]
        image, scale, offset_x, offset_y = image_padding(image)
        plt.imshow(image)
        for name, left, right, top, bottom in objects:
            left, right = left * scale + offset_x, right * scale + offset_x
            top, bottom = top * scale + offset_y, bottom * scale + offset_y
            #
            color = random.choice(['red', 'green', 'blue'])
            plt.text(left, top, name, color=color)
            rect = patches.Rectangle((left, top), (right - left), (bottom - top),
                                     edgecolor=color, linewidth=2, fill=False)
            ax.add_patch(rect)
        plt.pause(2.0)
        plt.close()


def prior_boxes():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
