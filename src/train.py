import argparse
from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf
from data_loader import mainz_recordloader, stanford_recordloader
from pathlib import Path
import math
import json
from utils import mean_metrics
#just for tests
#import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data', help="Directory with the TFRecord files.")
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-s', '--shuffle_size', default=1000, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-p', '--patience', default=10, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('--dataset', default='stanford', choices=['stanford', 'mainz'])
    args = parser.parse_args()

    train(args.batch_size, args.shuffle_size, args.epochs, args.patience, args.learning_rate, args.number_input_frames,
          Path(args.data_folder))


def train(batch_size, shuffle_size, epochs, patience, learning_rate, number_input_frames, data_folder):
    tf.random.set_seed(5)

    train_record_file_name = data_folder / 'tf_record' / 'train' / 'train_*.tfrecord.gzip'
    validation_record_file_name = data_folder / 'tf_record' / 'validation' / 'validation_*.tfrecord.gzip'
    metadata_path = data_folder / 'tf_record' / 'metadata.json'

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

    train_dataset = stanford_recordloader.build_dataset(str(train_record_file_name), batch_size, shuffle_size,
                                                    number_input_frames, shuffle=True)
    validation_dataset = stanford_recordloader.build_dataset_validation(str(validation_record_file_name),
                                                                        batch_size)

    # just for tests purposes
    #for test in train_set.take(1):
        #plt.imshow(test[0][0][10], cmap='gray')
        #plt.show()

    model = three_D_convolution_net.ThreeDConvolutionVGGStanford(width, height, number_input_frames, channels, mean,
                                                                 std)
    optimizer = keras.optimizers.Adam(learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    # opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn)


def train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn):
    early_stopping_counter = 0
    stop_training = False
    best_loss = math.inf
    train_mse_metric = keras.metrics.MeanSquaredError()
    validation_mse_metric = keras.metrics.MeanSquaredError()
    validation_mse_metric_distinct = keras.metrics.MeanSquaredError()
    validation_mse_metric_overlapping = keras.metrics.MeanSquaredError()

    Path("../logs").mkdir(exist_ok=True)
    Path("../saved").mkdir(exist_ok=True)
    log_dir = Path("../logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir_train = log_dir / 'train'
    log_dir_validation = log_dir / 'validation'
    save_path = Path("../saved, 'three_d_conv_best_model.h5'", datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer_train = tf.summary.create_file_writer(log_dir_train)
    file_writer_validation = tf.summary.create_file_writer(log_dir_validation)

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=(save_path / 'three_d_conv_best_model.h5'), monitor='val_loss',
                                        save_best_only=True, mode='min'),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    for epoch in range(epochs):
        for metric in (train_mse_metric, validation_mse_metric):
            metric.reset_states()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_mse_metric)

        with file_writer_train.as_default:
            tf.summary.scalar('MSE', data=train_mse_metric.result(), step=epoch)

        for x_batch_val, y_batch_val in validation_dataset:
            x_batch_val = x_batch_val[:, 0:50, :, :]
            validation_step(model, loss_fn, x_batch_val, y_batch_val, validation_mse_metric)
        validation_loss = validation_mse_metric.result()

        with file_writer_validation.as_default():
            tf.summary.scalar('MAE overlapping', data=validation_loss, step=epoch)

        # early stopping and save best model
        if validation_loss < best_loss:
            early_stopping_counter = 0
            best_loss = validation_loss
            model.save(save_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break
        # distinct_mae = mean_metrics.distinct_mean()
        # overlapping_mae = mean_metrics.overlapping_mean()
        # tf.summary.scalar('MAE distinct', data=distinct_mae, step=epoch)
        # tf.summary.scalar('MAE overlapping', data=overlapping_mae, step=epoch)

    # TODO implement swa


@tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_mse_metric):
    with tf.GradientTape() as tape:
        predictions = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, predictions)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_mse_metric.update_state(y_batch_train, predictions)
    return loss_value


@tf.function
def validation_step(model, loss_fn, x_validation, y_validation, validation_mse_metric):
    predictions = model(x_validation, training=False)
    mean_prediction = tf.reduce_mean(predictions)
    loss = loss_fn(y_validation, mean_prediction)
    validation_mse_metric.update_state(loss)


if __name__ == "__main__":
    # execute only if run as a script
    main()
