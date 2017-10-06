import chainer
import cv2
import numpy as np


class TestImageDataset(chainer.dataset.DatasetMixin):

    '''Resize, flip and crop only'''

    def __init__(self, paths, crop_size, mean=None):
        self.paths = paths
        self.crop_size = crop_size
        self.mean = mean

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        # load image
        full_path = self.paths[i]
        image = cv2.imread(full_path)  # BGR order
        h, w = image.shape[:2]

        if h < w:
            resize_h = max(256, self.crop_size)
            resize_w = w * resize_h // h
        else:
            resize_w = max(256, self.crop_size)
            resize_h = h * resize_w // w
        image = cv2.resize(image, (resize_w, resize_h))

        # Crop
        top = (resize_h - self.crop_size) // 2
        left = (resize_w - self.crop_size) // 2
        bottom = top + self.crop_size
        right = left + self.crop_size
        image = image[top:bottom, left:right, :]

        # Substract mean and transpose
        image = image.transpose(2, 0, 1).astype(np.float32)
        if self.mean is not None:
            image -= self.mean.astype(np.float32)

        return image
