import getopt
from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf
from data_loader import mainz_recordloader, stanford_recordloader
from pathlib import Path
import json
import sys
#import matplotlib.pyplot as plt


def main(argv):
    batch_size = 32
    shuffle_size = 1000
    epochs = 200
    patience = 20
    input_frames = 50
    learning_rate = 0.0008

    try:
        opts, args = getopt.getopt(argv, "b:s:e:p:l:", ["batch_size=", "shuffle_size=", "epochs=", "patience=",
                                                        "learning_rate="])
    except getopt.GetoptError:
        print('train.py -b <batch_size> -s <shuffle_size> -e <epochs> -e <epochs> -p <patience> -l <learning_rate>')
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
        elif opt in ("-l", "--learning_rate"):
            learning_rate = float(arg)
    train(batch_size, shuffle_size, epochs, patience, learning_rate, input_frames)


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
    main(sys.argv[1:])
