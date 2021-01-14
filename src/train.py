from utils import choose_gpu
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu.pick_gpu_lowest_memory())
#print("GPU:", str(choose_gpu.pick_gpu_lowest_memory()), 'will be used.')
from datetime import datetime
from models.three_D_vgg_net import ThreeDConvolutionVGG
from models.three_D_resnet import ThreeDConvolutionResNet18, ThreeDConvolutionResNet34, ThreeDConvolutionResNet50
from models.three_D_squeeze_and_excitation_resnet import ThreeDConvolutionSqueezeAndExciationResNet18
from data_loader import mainz_recordloader, stanford_recordloader
from visualisation import visualise
from pathlib import Path
import math
import utils.input_arguments
import json
import tensorflow as tf
from tensorflow import keras


# just for tests
# import matplotlib.pyplot as plt


def main():
    args = utils.input_arguments.get_train_arguments()
    train(args.batch_size, args.shuffle_size, args.epochs, args.patience, args.learning_rate, args.number_input_frames,
          Path(args.input_directory), args.dataset, args.model_name, args.experiment_name, args.augmentation,
          args.regularization, args.target)


def train(batch_size, shuffle_size, epochs, patience, learning_rate, number_input_frames, input_directory, dataset,
          model_name, experiment_name, augment, regularization, target):
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

    if dataset == 'mainz':
        train_dataset = mainz_recordloader.build_dataset(str(train_record_file_name), batch_size, shuffle_size,
                                                         number_input_frames, target, augment=augment)
        validation_dataset = mainz_recordloader.build_dataset_validation(str(validation_record_file_name), target)
    elif dataset == 'stanford':
        train_dataset = stanford_recordloader.build_dataset(str(train_record_file_name), batch_size, shuffle_size,
                                                            number_input_frames, augment=augment)
        validation_dataset = stanford_recordloader.build_dataset_validation(str(validation_record_file_name))

    # just for tests purposes
    # for test in train_set.take(1):
    # plt.imshow(test[0][0][10], cmap='gray')
    # plt.show()

    if model_name == 'resnet_18':
        model = ThreeDConvolutionResNet18(width, height, number_input_frames, channels, mean, std)
    elif model_name == 'resnet_34':
        model = ThreeDConvolutionResNet34(width, height, number_input_frames, channels, mean, std)
    elif model_name == 'resnet_50':
        model = ThreeDConvolutionResNet50(width, height, number_input_frames, channels, mean, std)
    elif model_name == 'se-resnet_18':
        model = ThreeDConvolutionSqueezeAndExciationResNet18(width, height, number_input_frames, channels, mean, std)
    else:
        model = ThreeDConvolutionVGG(width, height, number_input_frames, channels, mean, std)

    optimizer = keras.optimizers.Adam(learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn, number_input_frames,
               experiment_name, model_name, regularization)


def train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn, number_input_frames,
               experiment_name, model_name, regularization):
    early_stopping_counter = 0
    best_loss = math.inf
    train_mse_metric = keras.metrics.MeanSquaredError()
    train_mae_metric = keras.metrics.MeanAbsoluteError()
    train_metrics = [train_mse_metric, train_mae_metric]
    validation_mse_metric = keras.metrics.MeanSquaredError()
    validation_mae_metric = keras.metrics.MeanAbsoluteError()
    validation_mae_metric_distinct = keras.metrics.MeanAbsoluteError()
    validation_mae_metric_overlapping = keras.metrics.MeanAbsoluteError()

    Path("../logs").mkdir(exist_ok=True)
    Path("../saved").mkdir(exist_ok=True)
    log_dir = Path("../logs", '_'.join([experiment_name, model_name, datetime.now().strftime("%Y%m%d-%H%M%S")]))
    log_dir_train = log_dir / 'train'
    log_dir_validation = log_dir / 'validation'
    save_path = Path('../saved', '_'.join([experiment_name, model_name, datetime.now().strftime("%Y%m%d-%H%M%S")]))
    file_writer_train = tf.summary.create_file_writer(str(log_dir_train))
    file_writer_validation = tf.summary.create_file_writer(str(log_dir_validation))

    for epoch in range(epochs):
        # training
        for x_batch_train, y_batch_train in train_dataset.take(2):
            train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_metrics, regularization)

        with file_writer_train.as_default():
            tf.summary.scalar('epoch_loss', data=train_mse_metric.result(), step=epoch)
            tf.summary.scalar('epoch_mae', data=train_mae_metric.result(), step=epoch)

        # validation
        for x_batch_val, y_batch_val in validation_dataset.take(2):
            first_frames = get_first_frames(x_batch_val, number_input_frames)
            # distinct_splits = get_distinct_splits(x_batch_val, number_input_frames)
            # overlapping_splits = get_overlapping_splits(x_batch_val, number_input_frames)
            validation_step(model, first_frames, y_batch_val, validation_mse_metric)
            validation_step(model, first_frames, y_batch_val, validation_mae_metric)
            # validation_step(model, distinct_splits, y_batch_val, validation_mae_metric_distinct)
            # validation_step(model, overlapping_splits, y_batch_val, validation_mae_metric_overlapping)

        validation_mse = validation_mse_metric.result()
        validation_mae = validation_mae_metric.result()
        # validation_mae_overlapping = validation_mae_metric_overlapping.result()
        # validation_mae_distinct = validation_mae_metric_distinct.result()

        with file_writer_validation.as_default():
            tf.summary.scalar('epoch_loss', data=validation_mse, step=epoch)
            tf.summary.scalar('epoch_mae', data=validation_mae, step=epoch)
            # tf.summary.scalar('epoch_mae_overlapping', data=validation_mae_overlapping, step=epoch)
            # tf.summary.scalar('epoch_mae_distinct', data=validation_mae_distinct, step=epoch)

        for metric in (train_mse_metric, train_mae_metric, validation_mse_metric, validation_mae_metric,
                       validation_mae_metric_distinct, validation_mae_metric_overlapping):
            metric.reset_states()

        # early stopping and save best model
        if validation_mse < best_loss:
            early_stopping_counter = 0
            best_loss = validation_mse
            model.save_weights(str(save_path))
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break

    # visualization
    predictions = []
    true_values = []
    model.load_weights(save_path)
    for x_batch_val, y_batch_val in validation_dataset.take(2):
        first_frames = get_first_frames(x_batch_val, number_input_frames)
        prediction = validation_step(model, first_frames, y_batch_val, validation_mse_metric)
        predictions.append(prediction[0])
        true_values.append(y_batch_val[0])
    predictions = tf.concat(predictions, 0)
    true_values = tf.concat(true_values, 0)
    scatter_plot = visualise.create_scatter_plot(true_values, predictions)
    with file_writer_validation.as_default():
        tf.summary.image('Regression Plot', scatter_plot, step=0)


@tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, metrics, regularization):
    with tf.GradientTape() as tape:
        predictions = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, predictions)
        if regularization:
            loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for metric in metrics:
        metric.update_state(y_batch_train, predictions)
    return loss_value


@tf.function(experimental_relax_shapes=True)
def validation_step(model, x_validation, y_validation, validation_metric):
    predictions = model(x_validation, training=False)
    mean_prediction = tf.reduce_mean(predictions, keepdims=True)
    validation_metric.update_state(y_validation, mean_prediction)
    return mean_prediction


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
