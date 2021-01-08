import argparse
from datetime import datetime
from tensorflow import keras
from models import three_D_vgg_net
import tensorflow as tf
from data_loader import mainz_recordloader, stanford_recordloader
from pathlib import Path
import math
import json

# just for tests
# import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../data', help="Directory with the TFRecord files.")
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-s', '--shuffle_size', default=1000, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-p', '--patience', default=10, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('--dataset', default='stanford', choices=['stanford', 'mainz'])
    parser.add_argument('-m', '--model', default='vgg', choices=['vgg', 'resnet', 'se-resnet'])
    args = parser.parse_args()

    train(args.batch_size, args.shuffle_size, args.epochs, args.patience, args.learning_rate, args.number_input_frames,
          Path(args.input_directory), args.dataset, args.model)


def train(batch_size, shuffle_size, epochs, patience, learning_rate, number_input_frames, input_directory, dataset,
          model):
    tf.random.set_seed(5)

    train_record_file_name = input_directory / 'tf_record' / 'train' / 'train_*.tfrecord.gzip'
    validation_record_file_name = input_directory / 'tf_record' / 'validation' / 'validation_*.tfrecord.gzip'
    metadata_path = input_directory / 'tf_record' / 'metadata.json'

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
                                                        number_input_frames)
    validation_dataset = stanford_recordloader.build_dataset_validation(str(validation_record_file_name))

    # just for tests purposes
    # for test in train_set.take(1):
    # plt.imshow(test[0][0][10], cmap='gray')
    # plt.show()

    model = three_D_vgg_net.ThreeDConvolutionVGGStanford(width, height, number_input_frames, channels, mean,
                                                         std)
    optimizer = keras.optimizers.Adam(learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn, number_input_frames)


def train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn, number_input_frames):
    early_stopping_counter = 0
    best_loss = math.inf
    train_mse_metric = keras.metrics.MeanSquaredError()
    validation_mse_metric = keras.metrics.MeanSquaredError()
    validation_mae_metric = keras.metrics.MeanAbsoluteError()
    validation_mae_metric_distinct = keras.metrics.MeanAbsoluteError()
    validation_mae_metric_overlapping = keras.metrics.MeanAbsoluteError()

    Path("../logs").mkdir(exist_ok=True)
    Path("../saved").mkdir(exist_ok=True)
    log_dir = Path("../logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir_train = log_dir / 'train'
    log_dir_validation = log_dir / 'validation'
    save_path = Path('../saved', datetime.now().strftime("%Y%m%d-%H%M%S"), 'three_d_conv_best_model')
    file_writer_train = tf.summary.create_file_writer(str(log_dir_train))
    file_writer_validation = tf.summary.create_file_writer(str(log_dir_validation))

    for epoch in range(epochs):
        for metric in (train_mse_metric, validation_mse_metric, validation_mae_metric, validation_mae_metric_distinct,
                       validation_mae_metric_overlapping):
            metric.reset_states()

        for x_batch_train, y_batch_train in train_dataset.take(1):
            train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_mse_metric)

        with file_writer_train.as_default():
            tf.summary.scalar('epoch_loss', data=train_mse_metric.result(), step=epoch)

        for x_batch_val, y_batch_val in validation_dataset.take(1):
            first_frames = get_first_frames(x_batch_val, number_input_frames)
            distinct_splits = get_distinct_splits(x_batch_val, number_input_frames)
            overlapping_splits = get_overlapping_splits(x_batch_val, number_input_frames)
            validation_step(model, first_frames, y_batch_val, validation_mse_metric)
            validation_step(model, first_frames, y_batch_val, validation_mae_metric)
            validation_step(model, distinct_splits, y_batch_val, validation_mae_metric_distinct)
            validation_step(model, overlapping_splits, y_batch_val, validation_mae_metric_overlapping)

        validation_mse = validation_mse_metric.result()
        validation_mae = validation_mae_metric.result()
        validation_mae_overlapping = validation_mae_metric_overlapping.result()
        validation_mae_distinct = validation_mae_metric_distinct.result()

        with file_writer_validation.as_default():
            tf.summary.scalar('epoch_loss', data=validation_mse, step=epoch)
            tf.summary.scalar('epoch_mae', data=validation_mae, step=epoch)
            tf.summary.scalar('epoch_mae_overlapping', data=validation_mae_overlapping, step=epoch)
            tf.summary.scalar('epoch_mae_distinct', data=validation_mae_distinct, step=epoch)

        # early stopping and save best model
        if validation_mse < best_loss:
            early_stopping_counter = 0
            best_loss = validation_mse
            model.save(str(save_path), save_format="tf")
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break


@tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_mse_metric):
    with tf.GradientTape() as tape:
        predictions = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, predictions)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_mse_metric.update_state(y_batch_train, predictions)
    return loss_value


@tf.function(experimental_relax_shapes=True)
def validation_step(model, x_validation, y_validation, validation_metric):
    predictions = model(x_validation, training=False)
    mean_prediction = tf.reduce_mean(predictions, keepdims=True)
    validation_metric.update_state(y_validation, mean_prediction)


def get_distinct_splits(video, number_of_frames):
    number_of_subvideos = int(video.shape[1] / number_of_frames)
    subvideos = []
    for i in range(number_of_subvideos):
        start = i * number_of_frames
        end = start + number_of_frames
        subvideos.append(video[:, start: end:, :, :, :])
    return tf.concat(subvideos, 0)


def get_overlapping_splits(video, number_of_frames):
    number_of_subvideos = int(video.shape[1] / (number_of_frames / 2)) - 1
    half_number_of_frames = int(number_of_frames / 2)
    subvideos = []

    for i in range(number_of_subvideos):
        start = i * half_number_of_frames
        end = start + number_of_frames
        subvideos.append(video[:, start: end:, :, :, :])
    return tf.concat(subvideos, 0)


def get_first_frames(video, number_of_frames):
    return video[:, :number_of_frames, :, :, :]


if __name__ == "__main__":
    # execute only if run as a script
    main()
