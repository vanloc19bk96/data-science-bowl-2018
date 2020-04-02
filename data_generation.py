from dataset import Dataset
import tensorflow as tf
from functools import partial

def generate(root_dir):
    dataset = Dataset(root_dir)
    indices = dataset.ids

    for index in range(len(indices)):
        id = indices[index]
        image = dataset._get_image(id)
        label = dataset.label_encoding(id)
        yield id, image, label

def create_batch_generator(root_dir, batch_size):
    train_generator = partial(generate, root_dir)
    train_dataset = tf.data.Dataset.from_generator(train_generator, (tf.string, tf.float32, tf.uint8)).shuffle(40).batch(batch_size)
    return train_dataset.take(-1)

# create_batch_generator('data/stage1_train', 1)


