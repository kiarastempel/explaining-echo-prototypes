from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf
from data_loader import record_loader
from pathlib import Path

EPOCHS = 200


def train():
    train_record_file_name = '../data/dynamic-echo-data/tf_record/train.tfrecord'
    validation_record_file_name = '../data/dynamic-echo-data/tf_record/validation.tfrecord'

    model = three_D_convolution_net.ThreeDConvolution()
    opt = keras.optimizers.Adam(0.001)
    # opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    model.compile(
        optimizer=opt,
        loss=keras.losses.MeanSquaredError(),
        metrics=keras.metrics.MeanSquaredError()
    )
    log_dir = Path("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20),
        keras.callbacks.ModelCheckpoint(filepath='../saved/three_d_conv_best_model.h5', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    train_set = record_loader.build_dataset(train_record_file_name)
    validation_set = record_loader.build_dataset(validation_record_file_name)

    batch_size = 64
    model.fit(train_set, batch_size=batch_size, epochs=EPOCHS, callbacks=callbacks, validation_data=validation_set,
              verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    train()
