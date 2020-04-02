import os
import cv2
import tensorflow as tf
from utils import *

num_classes = 2
class Dataset(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.ids = os.listdir(self.root_dir)

    def _get_image(self, id):
        image_path = os.listdir(os.path.join(self.root_dir, id, 'images'))[0]
        image = cv2.imread(os.path.join(self.root_dir, id, 'images', image_path))
        image = cv2.resize(image, (256, 256))
        return tf.cast(image, tf.float32)

    def _get_masks(self, id):
        masks_path = os.listdir(os.path.join(self.root_dir, id, 'masks'))
        masks = [cv2.resize(cv2.imread(os.path.join(self.root_dir, id, 'masks', mask_path)), (256, 256)) for mask_path in masks_path]
        return masks

    def label_encoding(self, id):
        masks = self._get_masks(id)
        mask = combine_masks(masks)
        encoded = mask2onehot(mask, num_classes)
        return encoded
