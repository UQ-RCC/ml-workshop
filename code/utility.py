import datetime
import os
# Reduce default console output from tensorflow
# must be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from pathlib import Path
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def normalise(pixel):
    return pixel.astype('float32') / 255.0


def load_dataset():

    # Load dataset.
    (train_x, train_y), (validation_x, validation_y) = fashion_mnist.load_data()

    # Reshape dataset and normalise.
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    validation_x = validation_x.reshape((validation_x.shape[0], 28, 28, 1))
    train_x = normalise(train_x)
    validation_x = normalise(validation_x)

    # One hot encode target values.
    train_y = to_categorical(train_y)
    validation_y = to_categorical(validation_y)

    return train_x, train_y, validation_x, validation_y


def get_directory(*components):

    path = Path(*components)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_definition():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    return model


model_dir = get_directory('out', timestamp, 'model')
checkpoint_dir = get_directory('out', timestamp, 'checkpoint')
tb_log_dir = get_directory('out', timestamp, 'tb_log')

