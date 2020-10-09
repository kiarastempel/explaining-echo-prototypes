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


def main(argv):
    batch_size = 32
    shuffle_size = 1000
    epochs = 200

    try:
        opts, args = getopt.getopt(argv, "b:s:e:", ["batch_size=", "shuffle_size=", "epochs="])
    except getopt.GetoptError:
        print('train.py -b <batch_size> -s <shuffle_size> -e <epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        if opt in ("-s", "--shuffle_size"):
            shuffle_size = int(arg)
        if opt in ("-r", "--epochs"):
            epochs = int(arg)
    train(batch_size, shuffle_size, epochs)


def train(batch_size, shuffle_size, epochs):
    data_folder = Path('../data/dynamic-echo-data/tf_record/')
    train_record_file_name = data_folder / 'train' / 'train_*.tfrecord'
    validation_record_file_name = data_folder / 'validation' / 'validation_*.tfrecord'
    metadata_path = data_folder / 'metadata.json'

    with open(metadata_path) as metadata_file:
        metadata_json = json.load(metadata_file)
        metadata = metadata_json['metadata']
        width = metadata['frame_width']
        height = metadata['frame_height']
        number_of_frames = metadata['frame_count']
        number_of_test_samples = metadata['number_of_test_samples']
        number_of_train_samples = metadata['number_of_train_samples']
        number_of_validation_samples = metadata['number_of_validation_samples']
    channels = 1

    train_set = record_loader.build_dataset(str(train_record_file_name), batch_size, shuffle_size)
    validation_set = record_loader.build_dataset(str(validation_record_file_name), batch_size, shuffle_size)

    model = three_D_convolution_net.ThreeDConvolution_Stanford(width, height, number_of_frames, channels)
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
        keras.callbacks.ModelCheckpoint(filepath='../saved/three_d_conv_best_model.h5', monitor='val_mse',
                                        save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    model.fit(train_set, epochs=epochs, callbacks=callbacks, validation_data=validation_set,
              verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
