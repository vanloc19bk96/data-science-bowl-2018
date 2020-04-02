import numpy as np
import cv2
from skimage.morphology import label

def rle_decode(mask_rle):
    rows, cols = 256, 256
    rleNumbers = [int(numstring) for numstring in mask_rle.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    cv2.imshow("image", img)
    cv2.waitKey(0)
    return img

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 255)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return [str(run_lengths[i]) + " " + str(run_lengths[i + 1]) for i in range(0, len(run_lengths) - 1, 2)]

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def mask_to_rle(preds_test_upsampled, ids):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    return new_test_ids,rles