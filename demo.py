import cv2
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
from utils import *
from rle import *
from dataset import *
import pandas as pd
from skimage.transform import resize


image_size = 256
chanels = 3

model = tf.keras.models.load_model(r"C:\Users\Admin\Downloads\my_model.h5")

def submit():
    dataset = Dataset(r'data\stage1_test')
    ids = []
    sizes_test = []
    images = np.zeros((len(dataset.ids), 256, 256, 3), dtype=np.uint8)
    for n, id in enumerate(dataset.ids):
        image = np.array(dataset._get_image(id))
        sizes_test.append([image.shape[0], image.shape[1]])
        images[n] = image
        ids.append(id)
    predictions = model.predict(images)
    preds_test_upsampled = []
    for i in range(len(predictions)):
        preds_test_upsampled.append(resize(np.squeeze(predictions[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    test_ids, rles = mask_to_rle(preds_test_upsampled, dataset.ids)

    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018.csv', index=False)

if __name__ == '__main__':
    submit()