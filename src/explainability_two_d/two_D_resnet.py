from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from PIL import Image
from classification_models.keras import Classifiers
import tensorflow as tf
import torchvision.models as models
import random
import argparse
import pandas as pd
import cv2
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with the still images.')
    parser.add_argument('-o', '--output_directory',
                        help='Directory to save the model in.')
    parser.add_argument('-end', '--ending_out', default='',
                        help='Individual ending of output folder names')
    parser.add_argument('-fv', '--frame_volumes_filename', default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=2, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-l2', '--l2_regularization', default=0.01, type=float)
    parser.add_argument('-a', '--augmentation_intensity', default=10, type=int)
    parser.add_argument('-d', '--dropout_intensity', default=0.1, type=float)
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory)
    else:
        output_directory = Path(args.output_directory)
    output_directory_model = Path(output_directory, 'two_d_model_' + str(args.ending_out))
    output_directory_model.mkdir(parents=True, exist_ok=True)
    output_directory_history = Path(output_directory, 'metrics_history_' + str(args.ending_out))
    output_directory_history.mkdir(parents=True, exist_ok=True)
    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)

    tf.random.set_seed(5)
    random.seed(5)

    train(args.input_directory, frame_volumes_path, args.l2_regularization,
          args.dropout_intensity, args.augmentation_intensity,
          args.learning_rate, args.batch_size, args.epochs,
          output_directory_model, output_directory_history)


def train(input_directory, frame_volumes_path, l2_regularization,
          dropout_intensity, augmentation_intensity,
          learning_rate, batch_size, epochs,
          output_directory_model, output_directory_history):
    # get train/validation/test data
    train_still_images, train_volumes, _, test_still_images, test_volumes, _, val_still_images, val_volumes, _ = get_data(
        input_directory, frame_volumes_path, return_numpy_arrays=True)
    print('Data loaded')

    # get resnet model
    model = get_resnet18_model(l2_regularization, dropout_intensity)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse', 'mae', 'mape'])
    print('Compiled')

    # train model
    aug = augmentation_intensity
    train_data_gen = ImageDataGenerator(rotation_range=aug,
                                        width_shift_range=aug,
                                        height_shift_range=aug,
                                        brightness_range=(1 - float(aug)/100.0,
                                                          1 + float(aug)/100.0),
                                        rescale=1.0/255.0)
    train_data = train_data_gen.flow(train_still_images,
                                     train_volumes,
                                     batch_size=batch_size)
    # test if augmentation works
    # print(train_data)
    # for x, y in train_data:
    #     print(x[0])
    #     plt.imshow(x[0])
    #     print("y", y[0])
    #     plt.show()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.1, patience=20, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    history = model.fit(train_data,
                        validation_data=(val_still_images, val_volumes),
                        epochs=epochs,
                        callbacks=[tensorboard_callback, early_stop_callback])
    print('Fit')

    # save model
    model.save(output_directory_model)
    print('Model saved')

    # save metrics history
    save_history(history.history['mse'], output_directory_history, 'train_mse.txt')
    save_history(history.history['mae'], output_directory_history, 'train_mae.txt')
    save_history(history.history['mape'], output_directory_history, 'train_mape.txt')
    save_history(history.history['val_mse'], output_directory_history, 'val_mse.txt')
    save_history(history.history['val_mae'], output_directory_history, 'val_mae.txt')
    save_history(history.history['val_mape'], output_directory_history, 'val_mape.txt')
    print('History saved')

    # evaluate model
    loss, mse, mae, mape = model.evaluate(test_still_images, test_volumes)
    print('MSE Test:', mse)
    print('MAE Test:', mae)
    print('MAPE Test:', mape)
    save_history([mse, mae, mape], output_directory_history, 'eval_test.txt')
    loss, mse, mae, mape = model.evaluate(train_still_images, train_volumes)
    print('MSE Train:', mse)
    print('MAE Train:', mae)
    print('MAPE Train:', mape)
    save_history([mse, mae, mape], output_directory_history, 'eval_train.txt')


def get_data(input_directory, frame_volumes_path, return_numpy_arrays=False):
    train_still_images = []
    train_volumes = []
    train_filenames = []
    test_still_images = []
    test_volumes = []
    test_filenames = []
    val_still_images = []
    val_volumes = []
    val_filenames = []
    volume_tracings_data_frame = pd.read_csv(frame_volumes_path)

    i = 0
    for _, row in volume_tracings_data_frame.T.iteritems():
        print(i)
        i += 1
        # get image and y label
        image = Image.open(Path(input_directory, row['Image_FileName']))
        frame = np.asarray(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = frame / 255.0
        # plt.imshow(frame)
        # plt.show()
        if row['Split'] == 'TRAIN':
            train_still_images.append(frame)
            train_volumes.append(row['Volume'])
            train_filenames.append(row['Image_FileName'])
        elif row['Split'] == 'TEST':
            test_still_images.append(frame)
            test_volumes.append(row['Volume'])
            test_filenames.append(row['Image_FileName'])
        else:
            val_still_images.append(frame)
            val_volumes.append(row['Volume'])
            val_filenames.append(row['Image_FileName'])
    print("start converting")
    if return_numpy_arrays:
        train_still_images = np.array(train_still_images)
        train_volumes = np.array(train_volumes)
        test_still_images = np.array(test_still_images)
        test_volumes = np.array(test_volumes)
        val_still_images = np.array(val_still_images)
        val_volumes = np.array(val_volumes)
    return train_still_images, train_volumes, train_filenames, \
           test_still_images, test_volumes, test_filenames, \
           val_still_images, val_volumes, val_filenames


def get_resnet50_model(l2_reg):
    # somehow does not work
    # base_model = ResNet50(weights='imagenet', include_top=False,
    #                       input_tensor=Input(shape=(112, 112, 3)))
    # x = base_model.output
    # x = keras.layers.GlobalAvgPool2D()(x),
    # x = keras.layers.Flatten()(x),
    # x = keras.layers.Dense(units=1)(x),
    # model = keras.Model(inputs=base_model.input, outputs=x)

    model = keras.models.Sequential()
    model.add(ResNet50(include_top=False, input_tensor=Input(shape=(112, 112, 3))))  # weights=None
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dense(1))
    model.summary()

    add_regularization(model, regularizer=tf.keras.regularizers.l2(l2_reg))
    return model


def get_resnet18_model(l2_reg, dropout=0.1, untrained_layers=0):
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model = keras.models.Sequential()
    model.add(ResNet18(input_shape=(112, 112, 3), weights='imagenet',
                       include_top=False))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(1))

    # model = ResNet18(input_shape=(112, 112, 3), weights='imagenet',
    #                  include_top=False)
    # if untrained_layers != 0:
    #     for layer in model.layers[:-untrained_layers]:
    #         layer.trainable = False

    # x = keras.layers.GlobalAveragePooling2D()(model.output)
    # x = keras.layers.Dense(256)(x)
    # x = keras.layers.Dropout(dropout)(x)
    # output = keras.layers.Dense(1)(x)
    # model = keras.models.Model(inputs=[model.input], outputs=[output])
    model.summary()

    add_regularization(model, regularizer=tf.keras.regularizers.l2(l2_reg))
    return model


def get_resnet18_model_torch():
    model = models.resnet18(pretrained=True, progress=True)
    model.fc.out_features = 1
    model.train()
    # print(model.avgpool)
    # model = torch.nn.Sequential(*list(model.children())[:-1])
    # print(model)
    # add_regularization(model, regularizer=tf.keras.regularizers.l2(0.1))
    return model


# source: https://sthalles.github.io/keras-regularizer/
def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.01)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers[:-1]:
        for attr in ['kernel_regularizer']:
            # only add regularization to Conv2D-Layers
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When changing layers attributes, change only happens in model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def save_history(history, output_directory, file_name):
    out = Path(output_directory, '..', 'metrics_history')
    out.mkdir(parents=True, exist_ok=True)
    with open(Path(out, file_name), 'w') as txt_file:
        for h in history:
            txt_file.write(str(h) + '\n')


if __name__ == '__main__':
    main()

