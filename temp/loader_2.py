import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ssd.loader_1 import VOC2007, image_padding


def generate_prior_boxes_demo(image):
    # this function should be only called once at beginning
    #
    # feature map size: 32 * 32, 4 * 4
    # pixel size in origin: 8, 64
    # prior boxes size: 8 + 4 = 12, 64 + 32 = 96
    prior_boxes = []

    #
    fig, ax = plt.subplots()
    ax.imshow(image)

    # first mid point: (4, 4), size: 12, total num: 1024
    for y in range(3, 256, 8):
        for x in range(3, 256, 8):
            top, bottom, left, right = max(0, y - 6), min(256, y + 6), max(0, x - 6), min(256, x + 6)
            prior_boxes.append([top, bottom, left, right])
            # rect = patches.Rectangle((left, top), right - left, bottom - top,
            #                          edgecolor='red', linewidth=2, fill=False)
            # ax.add_patch(rect)

    # second mid point: (32, 32), size: 96, total num: 16
    colors = ['red', 'green', 'blue', 'pink', 'yellow']
    for y in range(31, 256, 64):
        for x in range(31, 256, 64):
            top, bottom, left, right = max(0, y - 48), min(256, y + 48), max(0, x - 48), min(256, x + 48)
            prior_boxes.append([top, bottom, left, right])
            rect = patches.Rectangle((left, top), right - left, bottom - top,
                                     edgecolor=colors[random.randrange(5)], linewidth=2, fill=False)
            ax.add_patch(rect)

    plt.show()


def generate_prior_boxes():
    prior_boxes = []

    # first mid point: (4, 4), size: 12, total num: 1024
    for y in range(3, 256, 8):
        for x in range(3, 256, 8):
            top, bottom, left, right = max(0, y - 6), min(256, y + 6), max(0, x - 6), min(256, x + 6)
            prior_boxes.append([top, bottom, left, right])

    # second mid point: (8, 8), size: 96, total num: 16
    for y in range(31, 256, 64):
        for x in range(31, 256, 64):
            top, bottom, left, right = max(0, y - 48), min(256, y + 48), max(0, x - 48), min(256, x + 48)
            prior_boxes.append([top, bottom, left, right])

    # pack them into array
    prior_boxes = np.array(prior_boxes, dtype=np.float32)

    return prior_boxes


def match(prior_boxes, boxes):
    # return the best overlapped prior_boxes index

    def intersect(box_1, box_2):
        # x_overlap = min(right_1, right_2) - max(left_1, left_2)
        x_overlap = np.minimum(box_1[:, 3], box_2[3]) - np.maximum(box_1[:, 2], box_2[2])
        x_overlap = np.maximum(0, x_overlap)

        # y_overlap = min(bottom_1, bottom_2) - max(top_1, top_2)
        y_overlap = np.minimum(box_1[:, 1], box_2[1]) - np.maximum(box_1[:, 0], box_2[0])
        y_overlap = np.maximum(0, y_overlap)

        return np.multiply(x_overlap, y_overlap)

    matched_idx = []
    for box in boxes:
        # compute overlapped position in prior_boxes
        area_overlap = intersect(prior_boxes, box)
        area_box = (box[3] - box[2]) * (box[1] - box[0])
        area_prior = np.multiply(prior_boxes[:, 3] - prior_boxes[:, 2], prior_boxes[:, 1] - prior_boxes[:, 0])
        jaccard = np.divide(area_overlap, (area_prior + area_box - area_overlap))

        # check
        assert np.alltrue(np.less(jaccard, 1.0))

        # save result
        idx = np.argmax(jaccard)
        matched_idx.append(idx)

    return matched_idx


def main():
    voc2007 = VOC2007()
    image, boxes = voc2007[666]
    image, boxes = image_padding(image, boxes, target_shape=(256, 256))

    # generate_prior_boxes_demo(image)
    prior_boxes = generate_prior_boxes()
    matched_idx = match(prior_boxes=prior_boxes, boxes=boxes)


if __name__ == '__main__':
    main()
