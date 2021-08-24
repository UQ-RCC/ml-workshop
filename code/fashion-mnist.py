# This code was adapted from the following source:
# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification

import os
# Reduce default console output from tensorflow
# must be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
import utility as utils


def compile_model(learning_rate, momentum):

    model = utils.get_model_definition()
    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():

    model_filename = str(utils.model_dir.joinpath('single-gpu'))
    checkpoint_filename = str(utils.checkpoint_dir.joinpath('checkpoint-{epoch}.h5'))
    tensorboard_log_dir = str(utils.tb_log_dir)

    verbose = True

    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32
    epochs = 10

    # Callbacks
    cb = list()

    cb.append(callbacks.ModelCheckpoint(checkpoint_filename))
    cb.append(callbacks.TensorBoard(log_dir=tensorboard_log_dir,
                                    histogram_freq=1))

    train_x, train_y, val_x, val_y = utils.load_dataset()
    model = compile_model(learning_rate, momentum)

    model.fit(train_x,
              train_y,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(val_x, val_y),
              callbacks=cb,
              verbose=verbose)

    model.save(model_filename)


if __name__ == '__main__':
    main()

