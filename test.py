import cv2
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
from utils import *
from rle import *
from dataset import *
import csv
from itertools import zip_longest
import pandas as pd
from skimage.transform import resize

image_size = 256
chanels = 3

model = tf.keras.models.load_model(r"C:\Users\Admin\Downloads\my_model_40.h5")
# def test(image):
#     # org = copy.deepcopy(image)
#     predict = model.predict(image.reshape(-1, image_size, image_size, 3))
#     encoded = rle_encode(onehot2mask(predict[0]))
#     return encoded
    # show_result(org, predict[0])

def show_result(image, result):
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.title.set_text('Actual image')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Predict truth labels')
    result = onehot2mask(result)
    ax2.imshow(result)
    plt.show()

def submit():
    dataset = Dataset(r"data/stage1_test")
    ids = []
    sizes_test = []
    images = np.zeros((len(dataset.ids), 256, 256, 3), dtype=np.uint8)
    for n, id in enumerate(dataset.ids):
        image = np.array(dataset._get_image(id))
        sizes_test.append([image.shape[0], image.shape[1]])
        images[n] = image
        ids.append(id)
    predictions = model.predict(images)
    preds_test_t = (predictions > 0.5).astype(np.uint8)
    # preds_test_upsampled = []
    # for i in range(len(predictions)):
    #     preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]),
    #                                        (sizes_test[i][0], sizes_test[i][1]),
    #                                        mode='constant', preserve_range=True))
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(ids):
        rle = rle_encoding(onehot2mask(predictions[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: x)
    sub.to_csv('sub-dsbowl2018-1.csv', index=False)

if __name__ == '__main__':
    submit()