import json
import math
import os
import random
import sys
import time
from pathlib import Path

import utils.input_arguments
from utils import choose_gpu
from utils import subvideos

os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu.pick_gpu_lowest_memory())
print("GPU:", str(choose_gpu.pick_gpu_lowest_memory()), 'will be used.')
from models.three_D_resnet import ThreeDConvolutionResNet18, ThreeDConvolutionResNet34, ThreeDConvolutionResNet50
from models.three_D_squeeze_and_excitation_resnet import ThreeDConvolutionSqueezeAndExciationResNet18
from data_loader import tf_record_loader
from visualisation import visualise
import tensorflow as tf
from tensorflow import keras


# just for tests
# import matplotlib.pyplot as plt


def main():
    args = utils.input_arguments.get_train_arguments()
    if args.resolution and len(args.resolution) == 1:
        resolution = (args.resolution[0], args.resolution[0])
    elif args.resolution and len(args.resolution) == 2:
        resolution = args.resolution
    elif args.resolution and len(args.resolution) > 2:
        sys.exit('Argument --resolution takes one or two values')
    else:
        resolution = None

    train(args.batch_size, args.shuffle_size, args.epochs, args.patience, args.learning_rate, args.number_input_frames,
          Path(args.input_directory), args.dataset, args.model_name, args.experiment_name, args.augmentation,
          args.regularization, args.target, args.load_checkpoint, resolution)


def train(batch_size, shuffle_size, epochs, patience, learning_rate, number_input_frames, input_directory, dataset,
          model_name, experiment_name, augment, regularization, target, load_checkpoint, resolution):
    # set random seeds for reproducibility
    tf.random.set_seed(5)
    random.seed(5)

    train_record_file_name = input_directory / 'train' / 'train_*.tfrecord.gzip'
    validation_record_file_name = input_directory / 'validation' / 'validation_*.tfrecord.gzip'
    metadata_path = input_directory / 'metadata.json'

    with open(metadata_path) as metadata_file:
        metadata_json = json.load(metadata_file)
        metadata = metadata_json['metadata']
        width = metadata['frame_width']
        height = metadata['frame_height']
        mean = metadata['mean']
        std = metadata['std']
        channels = metadata['channels']

    train_dataset = tf_record_loader.build_dataset(str(train_record_file_name), batch_size, shuffle_size,
                                                   number_input_frames, resolution, augment, dataset, target)
    validation_dataset = tf_record_loader.build_dataset(str(validation_record_file_name), batch_size,
                                                        None, number_input_frames, resolution, False, dataset, target)
    mean_validation_dataset = tf_record_loader.build_dataset(str(validation_record_file_name), 1, None,
                                                             number_input_frames, resolution, False, dataset, target,
                                                             full_video=True)

    # for batch in train_dataset.take(1):
    # for i in range(number_input_frames):
    # plt.imshow(batch[0][0][i], cmap='gray')
    # plt.show()
    output = 2 if target == 'all' else 1

    if model_name == 'resnet_18':
        model = ThreeDConvolutionResNet18(width, height, number_input_frames, channels, mean, std, output,
                                          regularization)
    elif model_name == 'resnet_34':
        model = ThreeDConvolutionResNet34(width, height, number_input_frames, channels, mean, std, output,
                                          regularization)
    elif model_name == 'resnet_50':
        model = ThreeDConvolutionResNet50(width, height, number_input_frames, channels, mean, std, output,
                                          regularization)
    elif model_name == 'se-resnet_18':
        model = ThreeDConvolutionSqueezeAndExciationResNet18(width, height, number_input_frames, channels, mean, std,
                                                             output, regularization)
    else:
        model = ThreeDConvolutionResNet34(width, height, number_input_frames, channels, mean, std, output,
                                          regularization)

    optimizer = keras.optimizers.Adam(learning_rate)
    loss_fn = keras.losses.MeanSquaredError()

    # benchmark(train_dataset)
    save_name = '_'.join([experiment_name, model_name, dataset, target])
    train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn, number_input_frames,
               save_name, load_checkpoint, mean_validation_dataset, target, dataset)


def train_loop(model, train_dataset, validation_dataset, patience, epochs, optimizer, loss_fn, number_input_frames,
               save_name, load_checkpoint, mean_validation_dataset, target, dataset):
    start_epoch = 0
    checkpoint = tf.train.Checkpoint(step_counter=tf.Variable(0), optimizer=optimizer, net=model,
                                     iterator=train_dataset)
    checkpoint_path = Path('./tf_checkpoints', save_name)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=2)

    if load_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        start_epoch = int(checkpoint.step_counter + 1)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
    early_stopping_counter = 0
    best_loss = math.inf
    train_mse_metric = keras.metrics.MeanSquaredError()
    train_mae_metric = keras.metrics.MeanAbsoluteError()
    train_metrics = [train_mse_metric, train_mae_metric]
    validation_mse_metric = keras.metrics.MeanSquaredError()
    validation_mae_metric = keras.metrics.MeanAbsoluteError()

    Path("../logs").mkdir(exist_ok=True)
    Path("../saved").mkdir(exist_ok=True)

    log_dir = Path("../logs", save_name)
    log_dir_train = log_dir / 'train'
    log_dir_validation = log_dir / 'validation'
    save_path = Path('../saved', save_name)
    file_writer_train = tf.summary.create_file_writer(str(log_dir_train))
    file_writer_validation = tf.summary.create_file_writer(str(log_dir_validation))

    for epoch in range(start_epoch, epochs):
        # training
        for x_batch_train, y_batch_train in train_dataset.take(1):
            train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_metrics)

        with file_writer_train.as_default():
            tf.summary.scalar('epoch_loss', data=train_mse_metric.result(), step=epoch)
            tf.summary.scalar('epoch_mae', data=train_mae_metric.result(), step=epoch)

        # validation
        for x_batch_val, y_batch_val in validation_dataset.take(1):
            val_predictions = model(x_batch_val, training=False)
            validation_mse_metric.update_state(y_batch_val, val_predictions)
            validation_mae_metric.update_state(y_batch_val, val_predictions)

        validation_mse = validation_mse_metric.result()
        with file_writer_validation.as_default():
            tf.summary.scalar('epoch_loss', data=validation_mse, step=epoch)
            tf.summary.scalar('epoch_mae', data=validation_mae_metric.result(), step=epoch)

        # checkpoint
        checkpoint.step_counter.assign_add(1)
        # save every after every third epoch
        if epoch % 3 == 0:
            manager.save()

        for metric in (train_mse_metric, train_mae_metric, validation_mse_metric, validation_mae_metric):
            metric.reset_states()

        # reduce lr on plateau and early stopping with save best model
        if validation_mse < best_loss:
            early_stopping_counter = 0
            best_loss = validation_mse
            model.save_weights(str(save_path))
        else:
            early_stopping_counter += 1
            if early_stopping_counter > int(patience / 2):
                optimizer.lr.assign(optimizer.lr.read_value() / 2)
            if early_stopping_counter > patience:
                break
    print(f'{best_loss=}')
    validation_mae_metric_distinct = keras.metrics.MeanAbsoluteError()
    validation_mae_metric_overlapping = keras.metrics.MeanAbsoluteError()
    # visualization
    predictions_distinct = []
    predictions_overlapping = []
    true_values = []
    for x_batch_val, y_batch_val in mean_validation_dataset.take(5):
        true_values.append(y_batch_val)
        distinct_splits = subvideos.get_distinct_splits(x_batch_val, number_input_frames)
        overlapping_splits = subvideos.get_overlapping_splits(x_batch_val, number_input_frames)
        mean_prediction_distinct = validation_step(model, distinct_splits, y_batch_val, validation_mae_metric_distinct)
        mean_prediction_overlapping = validation_step(model, overlapping_splits, y_batch_val,
                                                      validation_mae_metric_overlapping)
        predictions_distinct.append(tf.squeeze(mean_prediction_distinct))
        predictions_overlapping.append(tf.squeeze(mean_prediction_overlapping))
    predictions_distinct = tf.concat(predictions_distinct, 0)
    predictions_overlapping = tf.concat(predictions_overlapping, 0)
    true_values = tf.concat(true_values, 0)
    scatter_plot_distinct = visualise.create_scatter_plot(true_values, predictions_distinct, target, dataset)
    scatter_plot_overlapping = visualise.create_scatter_plot(true_values, predictions_overlapping, target, dataset)

    with file_writer_validation.as_default():
        tf.summary.scalar('epoch_mae_overlapping', data=validation_mae_metric_overlapping.result(), step=epochs)
        tf.summary.scalar('epoch_mae_distinct', data=validation_mae_metric_distinct.result(), step=epochs)
        tf.summary.image('Regression Plot', scatter_plot_distinct, step=0)
        tf.summary.image('Regression Plot', scatter_plot_overlapping, step=0)

    # visualization
    predictions = []
    true_values = []
    model.load_weights(str(save_path))
    for x_batch_val, y_batch_val in validation_dataset:
        prediction = model(x_batch_val, training=False)
        predictions.append(tf.squeeze(prediction))
        true_values.append(y_batch_val)
    predictions = tf.concat(predictions, 0)
    true_values = tf.concat(true_values, 0)
    scatter_plot = visualise.create_scatter_plot(true_values, predictions, target, dataset)
    with file_writer_validation.as_default():
        tf.summary.image('Regression Plot', scatter_plot, step=0)

    model.save(str(save_path))


@tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, metrics):
    with tf.GradientTape() as tape:
        predictions = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, predictions)
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


def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for _, _ in dataset.take(10):
            # Performing a training step
            pass
    print("Execution time:", time.perf_counter() - start_time)


if __name__ == "__main__":
    # execute only if run as a script
    main()
