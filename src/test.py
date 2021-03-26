from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import utils.input_arguments
from data_loader import tf_record_loader
from utils import subvideos, visualise


def main():
    args = utils.input_arguments.get_test_arguments()
    test(args.number_input_frames, args.dataset, args.model_path, args.input_directory, args.target, args.batch_size,
         args.resolution)


def test(number_input_frames, dataset, model_path, input_directory, target, batch_size, resolution):
    data_folder = Path(input_directory)
    test_record_file_names = data_folder / 'test' / 'test_*.tfrecord.gzip'

    model = keras.models.load_model(model_path)

    test_dataset = tf_record_loader.build_dataset(str(test_record_file_names), batch_size, None, number_input_frames,
                                                  resolution, False, dataset, target, False)
    mean_test_dataset = tf_record_loader.build_dataset(str(test_record_file_names), 1, None,
                                                       number_input_frames, resolution, False, dataset, target,
                                                       full_video=True)

    test_mse_metric = keras.metrics.MeanSquaredError()
    test_mae_metric = keras.metrics.MeanAbsoluteError()

    # test
    for x_batch_val, y_batch_val in test_dataset:
        test_step(model, x_batch_val, y_batch_val, test_mse_metric)
        test_step(model, x_batch_val, y_batch_val, test_mae_metric)

    validation_mae_metric_distinct = keras.metrics.MeanAbsoluteError()
    validation_mae_metric_overlapping = keras.metrics.MeanAbsoluteError()
    for x_batch_val, y_batch_val in mean_test_dataset:
        distinct_splits = subvideos.get_distinct_splits(x_batch_val, number_input_frames)
        overlapping_splits = subvideos.get_overlapping_splits(x_batch_val, number_input_frames)
        validation_step(model, distinct_splits, y_batch_val, validation_mae_metric_distinct)
        validation_step(model, overlapping_splits, y_batch_val, validation_mae_metric_overlapping)

    print('epoch_mae_overlapping', validation_mae_metric_overlapping.result().numpy())
    print('epoch_mae_distinct', validation_mae_metric_distinct.result().numpy())
    print("Test MSE:", test_mse_metric.result().numpy())
    print("Test MAE:", test_mae_metric.result().numpy())

    predictions = []
    true_values = []
    for x_batch_val, y_batch_val in test_dataset:
        prediction = model(x_batch_val, training=False)
        predictions.append(tf.squeeze(prediction))
        true_values.append(y_batch_val)
    predictions = tf.concat(predictions, 0)
    true_values = tf.concat(true_values, 0)
    scatter_plot = visualise.create_scatter_plot(true_values, predictions)
    tf.keras.preprocessing.image.save_img('scatter_plot.png', scatter_plot)


@tf.function(experimental_relax_shapes=True)
def test_step(model, x_validation, y_validation, validation_metric):
    predictions = model(x_validation, training=False)
    mean_prediction = tf.reduce_mean(predictions, keepdims=True)
    validation_metric.update_state(y_validation, mean_prediction)
    return mean_prediction


@tf.function(experimental_relax_shapes=True)
def validation_step(model, x_validation, y_validation, validation_metric):
    predictions = model(x_validation, training=False)
    mean_prediction = tf.reduce_mean(predictions, keepdims=True)
    validation_metric.update_state(y_validation, mean_prediction)
    return mean_prediction


if __name__ == "__main__":
    # execute only if run as a script
    main()
