import argparse
from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf
from data_loader import mainz_recordloader, stanford_recordloader
from pathlib import Path
import json
#import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('b', 'batch_size', required=False, default=32, type=int)
    parser.add_argument('s', 'shuffle_size', default=1000, type=int)
    parser.add_argument('e', 'epochs', default=200, type=int)
    parser.add_argument('p', 'patience', default=20, type=int)
    parser.add_argument('l', 'learning_rate', default=0.008, type=float)
    parser.add_argument('f', 'number_input_frames', default=50, type=int)
    args = parser.parse_args()

    train(args.batch_size, args.shuffle_size, args.epochs, args.patience, args.learning_rate, args.input_frames)


def train(batch_size, shuffle_size, epochs, patience, learning_rate, input_frames):
    tf.random.set_seed(5)

    data_folder = Path('../data/dynamic-echo-data/tf_record/')
    train_record_file_name = data_folder / 'train' / 'train_*.tfrecord'
    validation_record_file_name = data_folder / 'validation' / 'validation_*.tfrecord'
    metadata_path = data_folder / 'metadata.json'

    with open(metadata_path) as metadata_file:
        metadata_json = json.load(metadata_file)
        metadata = metadata_json['metadata']
        width = metadata['frame_width']
        height = metadata['frame_height']
        number_of_test_samples = metadata['number_of_test_samples']
        number_of_train_samples = metadata['number_of_train_samples']
        number_of_validation_samples = metadata['number_of_validation_samples']
        mean = metadata['mean']
        std = metadata['std']
        channels = metadata['channels']

    train_set = stanford_recordloader.build_dataset(str(train_record_file_name), batch_size, shuffle_size, input_frames,
                                            split=False)
    validation_set = stanford_recordloader.build_dataset(str(validation_record_file_name), batch_size, shuffle_size,
                                                 input_frames)

     #for test in train_set.take(1):
        #print(test)
        # plt.imshow(test[0][0][10], cmap='gray')
        # plt.show()

    model = three_D_convolution_net.ThreeDConvolution_Stanford(width, height, input_frames, channels, mean, std)
    opt = keras.optimizers.Adam(learning_rate)
    # opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    Path("../logs").mkdir(exist_ok=True)
    log_dir = Path("../logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path("../saved").mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint(filepath=(log_dir / 'three_d_conv_best_model.h5'), monitor='val_loss',
                                        save_best_only=True, mode='min'),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    model.fit(train_set, epochs=epochs, callbacks=callbacks, validation_data=validation_set,
              verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    main()
