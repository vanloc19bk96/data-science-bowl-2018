from data_generation import *
from model import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, BinaryAccuracy

image_size = 256
chanels = 3
train_path = 'data/stage1_train'
batch_size = 10
epochs = 10
optimizer = Adam()
loss = BinaryCrossentropy()
epoch_loss_avg = Mean()
epoch_accuracy = BinaryAccuracy()

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        loss_value = loss(y_true=targets, y_pred=prediction)
        print(loss_value)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradients

if __name__ == '__main__':
    batch_generator = create_batch_generator(train_path, batch_size)
    model = build_unet(image_size, chanels)
    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(epochs):
        for i, (id, image, label) in enumerate(batch_generator):
            loss_value, grads = grad(model, image, label)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg(loss_value)
            epoch_accuracy(label, model(image))
        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))