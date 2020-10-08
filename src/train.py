import getopt
from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf
from data_loader import record_loader
from pathlib import Path
import json
import sys


EPOCHS = 200


def main(argv):
    batch_size = 32
    try:
        opts, args = getopt.getopt(argv, "b:", ["batch_size="])
    except getopt.GetoptError:
        print('test.py -b <batch_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-b", "--batch_size"):
            batch_size = int(batch_size)
    train(batch_size)


def train(batch_size):
    data_folder = Path('../data/dynamic-echo-data/tf_record/')
    train_record_file_name = data_folder / 'train' / 'train_*.tfrecord'
    validation_record_file_name = data_folder / 'validation' / 'validation_*.tfrecord'
    metadata_path = data_folder / 'metadata.json'
    train_set = record_loader.build_dataset(str(train_record_file_name), batch_size)
    validation_set = record_loader.build_dataset(str(validation_record_file_name), batch_size)

    with open(metadata_path) as metadata_file:
        metadata_json = json.load(metadata_file)
        metadata = metadata_json['metadata']
        width = metadata['frame_width']
        height = metadata['frame_height']
        number_of_frames = metadata['frame_count']
    channels = 1

    model = three_D_convolution_net.ThreeDConvolution(width, height, number_of_frames, channels)
    opt = keras.optimizers.Adam(0.001)
    # opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    model.compile(
        optimizer=opt,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    log_dir = Path("../logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20),
        keras.callbacks.ModelCheckpoint(filepath='../saved/three_d_conv_best_model.h5', monitor='val_mse', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    model.fit(train_set, epochs=EPOCHS, callbacks=callbacks, validation_data=validation_set,
              verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
