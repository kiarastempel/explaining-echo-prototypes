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
    patience = 20

    try:
        opts, args = getopt.getopt(argv, "b:s:e:p:", ["batch_size=", "shuffle_size=", "epochs=", "patience="], )
    except getopt.GetoptError:
        print('train.py -b <batch_size> -s <shuffle_size> -e <epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-s", "--shuffle_size"):
            shuffle_size = int(arg)
        elif opt in ("-r", "--epochs"):
            epochs = int(arg)
        elif opt in ("-p", "--patience"):
            patience = int(arg)
    train(batch_size, shuffle_size, epochs, patience)


def train(batch_size, shuffle_size, epochs, patience):
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
        mean = metadata['mean']
        std = metadata['std']
        channels = metadata['channels']

    train_set = record_loader.build_dataset(str(train_record_file_name), batch_size, shuffle_size, augment=True)
    validation_set = record_loader.build_dataset(str(validation_record_file_name), batch_size, shuffle_size)

    model = three_D_convolution_net.ThreeDConvolution_Stanford(width, height, number_of_frames, channels, mean, std)
    opt = keras.optimizers.Adam(0.001)
    # opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    Path("../logs").mkdir(exist_ok=True)
    log_dir = Path("../logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path("../saved").mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint(filepath='../saved/three_d_conv_best_model.h5', monitor='val_loss',
                                        save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    model.fit(train_set, epochs=epochs, callbacks=callbacks, validation_data=validation_set,
              verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
