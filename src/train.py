from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf
from data_loader import record_loader
from pathlib import Path
import json
import cv2


EPOCHS = 200


def train():
    data_folder = Path('../data/dynamic-echo-data/tf_record/')
    train_record_file_name = data_folder / 'train' / 'train_*.tfrecord'
    validation_record_file_name = data_folder / 'validation' / 'validation_*.tfrecord'
    metadata_path = data_folder / 'metadata.json'
    train_set = record_loader.build_dataset(str(train_record_file_name))
    validation_set = record_loader.build_dataset(str(validation_record_file_name))

    # for sample in train_set.take(1):
    #     pic = sample[0][0]
    #     for frame in pic:
    #         test = (frame.numpy())
    #         cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    #         cv2.imshow('dst_rt', frame.numpy())
    #         cv2.waitKey(0)
    #     cv2.destroyAllWindows()

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
        metrics=keras.metrics.MeanSquaredError()
    )
    log_dir = Path("../logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20),
        keras.callbacks.ModelCheckpoint(filepath='../saved/three_d_conv_best_model.h5', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    batch_size = 64
    model.fit(train_set, batch_size=batch_size, epochs=EPOCHS, callbacks=callbacks, validation_data=validation_set,
              verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    train()
