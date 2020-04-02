import numpy as np

def combine_masks(masks):
    return np.bitwise_or.reduce(masks)

def mask2onehot(mask, num_classes):
    shape = mask.shape[:2] + (num_classes, )
    encoded = np.zeros(shape, dtype=np.uint8)
    indexes = np.where(np.all(mask[:, :] == [255, 255, 255], axis=2))
    indexes = zip(indexes[0], indexes[1])
    for index in indexes:
        encoded[index[0], index[1]] = 1
    return encoded

def onehot2mask(onehot):
    num_chanels = 3
    output = np.zeros(onehot.shape[:2] + (num_chanels,))
    indexes = np.where(np.all(onehot[:, :] > 0.5, axis=2))
    indexes = zip(indexes[0], indexes[1])
    for index in indexes:
        output[index[0], index[1]] = [255, 255, 255]
    return output