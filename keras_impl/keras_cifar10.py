import tensorflow as tf
from tensorflow import keras
import numpy as np


class KerasLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        # print a dot to show progress
        if batch % 100 == 0:
            print('Epoch: {}'.format(batch))
        print('>', end='')


class KerasCifar10:

    def __init__(self, learning_rate=0.001, validation_split=0.2, batch_size=32, epochs=50,
                 log_dir=None):
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir

    def get_data(self):
        # Load training and eval data
        print("Load training and eval data")
        (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()

        # do not use tf.cast
        train_data = np.asarray(train_data, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.int32)
        test_data = np.asarray(test_data, dtype=np.float32)
        test_labels = np.asarray(test_labels, dtype=np.int32)

        print("test_data Shape after padding: {}".format(test_data.shape))
        print("train_data Shape after padding: {}".format(train_data.shape))
        tf.summary.image("input_train", train_data)
        tf.summary.image("input_test", test_data)
        return (train_data, train_labels), (test_data, test_labels)

    def execute_model(self, get_model, qn):
        # Get data
        with tf.device('/cpu:0'):
            (train_data, train_labels), (eval_data, eval_labels) = self.get_data()
            # one-hot encoding for the labels
            train_labels = keras.utils.to_categorical(train_labels)
            eval_labels = keras.utils.to_categorical(eval_labels)
            train_data /= 255
            eval_data /= 255

        # Build model
        model = get_model(train_data[0].shape)
        model.summary()

        # Compile model
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.SGD(lr=self.learning_rate)  # , decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # Train model
        if self.log_dir is not None:
            checkpoint_folder = self.log_dir + "/" + qn
        else:
            checkpoint_folder = "./" + qn

        last_epoch_ran = 0
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_folder)
        if latest_checkpoint_path is not None:
            print("\tRestoring checkpoint from %s" % latest_checkpoint_path)
            model.load_weights(latest_checkpoint_path)
            end_i = latest_checkpoint_path.find(".ckpt")
            last_epoch_ran = int(latest_checkpoint_path[end_i-4:end_i])
            print("\tInitial epoch: %d" % last_epoch_ran)

        checkpoint_path = checkpoint_folder + "/train-{epoch:04d}.ckpt"
        save_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        callbacks = [save_checkpoint]

        history = model.fit(train_data, train_labels,
                            batch_size=self.batch_size,
                            epochs=self.epochs, initial_epoch=last_epoch_ran,
                            verbose=1, shuffle=True,
                            validation_split=self.validation_split,
                            callbacks=callbacks)

        # Test model
        test_loss, test_acc = model.evaluate(eval_data, eval_labels, verbose=1)

        print("test_loss: {}. test_acc: {}".format(test_loss, test_acc))

        # clear memory
        keras.backend.clear_session()

        return history, test_loss, test_acc
