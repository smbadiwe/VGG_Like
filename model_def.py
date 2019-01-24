import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def conv_2d(filters, input_layer_shape=None, name=None):
    if input_layer_shape:
        return keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                                   activation=tf.nn.relu, input_shape=input_layer_shape,
                                   padding="same", name=name)

    return keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                               activation=tf.nn.relu,
                               padding="same", name=name)


def pooling(name=None):
    return keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name=name)


def build_model(input_layer_shape, qn2=False, qn3=False):
    """
    VGG-like CNN model definition. This returns the model
    :param input_layer_shape:
    :param qn2: If true, then we're doing Qn.2
    :param qn3: If true, then we're doing Qn.3
    :return:
    """
    print("input_data Shape: {}".format(input_layer_shape))
    model = keras.Sequential()

    # block 1
    model.add(conv_2d(64, input_layer_shape))
    model.add(conv_2d(64))
    model.add(pooling())

    if not qn3:
        # block 2
        model.add(conv_2d(128))
        model.add(conv_2d(128))
        model.add(pooling())

    if not qn2:
        # block 3
        model.add(conv_2d(256))
        model.add(conv_2d(256))
        model.add(pooling())

    # block 4
    model.add(conv_2d(512))
    model.add(conv_2d(512))

    model.add(keras.layers.Flatten())

    # Dense Layers
    model.add(keras.layers.Dense(200, activation=tf.nn.relu))
    model.add(keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax, name="Softmax"))

    return model


def problem_1(input_layer_shape):
    return build_model(input_layer_shape)


def problem_2(input_layer_shape):
    return build_model(input_layer_shape, qn2=True)


def problem_3(input_layer_shape):
    return build_model(input_layer_shape, qn3=True)


def plot_loss(epochs, history1, history2, history3):
    plt.subplot(2, 2, 1)
    plt.plot(history1.history['acc'])
    plt.plot(history2.history['acc'])
    plt.plot(history3.history['acc'])

    plt.xticks(np.arange(0, epochs, (epochs // 10)))
    plt.title('Train accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)

    plt.subplot(2, 2, 2)
    plt.plot(history1.history['loss'])
    plt.plot(history2.history['loss'])
    plt.plot(history3.history['loss'])

    plt.xticks(np.arange(0, epochs, (epochs // 10)))
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)

    plt.subplot(2, 2, 3)
    plt.plot(history1.history['val_acc'])
    plt.plot(history2.history['val_acc'])
    plt.plot(history3.history['val_acc'])

    plt.xticks(np.arange(0, epochs, (epochs // 10)))
    plt.title('Val accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)

    plt.subplot(2, 2, 4)
    plt.plot(history1.history['val_loss'])
    plt.plot(history2.history['val_loss'])
    plt.plot(history3.history['val_loss'])

    plt.xticks(np.arange(0, epochs, (epochs / 10)))
    plt.title('Val loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)

    plt.tight_layout()
    plt.show()


def run():
    from keras_cifar10 import KerasCifar10

    c = KerasCifar10()
    history1, test_loss1, test_acc1 = c.execute_model(problem_1)
    history2, test_loss2, test_acc2 = c.execute_model(problem_2)
    history3, test_loss3, test_acc3 = c.execute_model(problem_3)

    print("Q1: Test Acc: {}. Test Loss: {}".format(test_acc1, test_loss1))
    print("Q2: Test Acc: {}. Test Loss: {}".format(test_acc2, test_loss2))
    print("Q3: Test Acc: {}. Test Loss: {}".format(test_acc3, test_loss3))
    
    plot_loss(c.epochs, history1, history2, history3)


if __name__ == "__main__":
    run()