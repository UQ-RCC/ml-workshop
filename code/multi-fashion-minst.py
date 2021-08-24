# This code was adapted from the following source:
# source https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification

import os
# Reduce default console output from tensorflow
# must be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import utility as utils

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# define cnn model
def compile_model(learning_rate, momentum):

    model = utils.get_model_definition()

    # compile model with Horovod DistributedOptimizer.
    # opt = SGD(lr=learning_rate, momentum=momentum)
    opt = hvd.DistributedOptimizer(SGD(lr=learning_rate, momentum=momentum))

    # Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses Horovod's DistributedOptimizer to compute gradients.
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model


def main():

    model_filename = str(utils.model_dir.joinpath('multi-gpu'))
    checkpoint_filename = str(utils.checkpoint_dir.joinpath('checkpoint-{epoch}.h5'))
    tensorboard_log_dir = str(utils.tb_log_dir)

    verbose = False

    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32
    epochs = 10

    # Increase the batch size to decrease network traffic between GPUs
    batch_size *= hvd.size()
    # Scale the learning rate (SGD Optimisers benefit from this).
    # source: Accurate, Large Minibatch SGD Training ImageNet in 1 Hour
    learning_rate *= hvd.size()

    # Callbacks
    cb = list()

    # Broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    cb.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    cb.append(hvd.callbacks.MetricAverageCallback())
    cb.append(hvd.callbacks.LearningRateWarmupCallback(initial_lr=learning_rate / hvd.size(),
                                                       warmup_epochs=3,
                                                       verbose=0))

    # Save checkpoint and tensorboard files on worker 0 only.
    # The model is identical across all workers; therefore, other workers should not
    # save this information. Concurrent I/O could also corrupt output files.
    if hvd.rank() == 0:
        cb.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_filename))
        cb.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,
                                                 histogram_freq=1))
    # Other workers should not write to the console.
    else:
        verbose = False

    train_x, train_y, val_x, val_y = utils.load_dataset()
    model = compile_model(learning_rate, momentum)

    # All workers compute the model together.
    model.fit(train_x,
              train_y,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(val_x, val_y),
              callbacks=cb,
              verbose=verbose)

    # Worker 0 saves the model when complete (model identical across all workers).
    if hvd.rank() == 0:
        model.save(model_filename)


if __name__ == "__main__":
    main()
