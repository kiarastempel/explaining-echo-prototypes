from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import utils.input_arguments
from data_loader import tf_record_loader


def main():
    args = utils.input_arguments.get_test_arguments()
    test(args.number_input_frames, args.dataset, args.model_path, args.input_directory, args.target)


def test(number_input_frames, dataset, model_path, input_directory, target):
    data_folder = Path(input_directory)
    test_record_file_names = data_folder / 'test' / 'test_*.tfrecord.gzip'

    model = keras.models.load_model(model_path)

    test_dataset = tf_record_loader.build_dataset(str(test_record_file_names), None, 1, number_input_frames, False,
                                                  dataset, target)

    test_mse_metric = keras.metrics.MeanSquaredError()
    test_mae_metric = keras.metrics.MeanAbsoluteError()

    # test
    for x_batch_val, y_batch_val in test_dataset:
        first_frames = get_first_frames(x_batch_val, number_input_frames)
        test_step(model, first_frames, y_batch_val, test_mse_metric)
        test_step(model, first_frames, y_batch_val, test_mae_metric)

    print("Test MSE:", test_mse_metric.result().numpy())
    print("Test MAE:", test_mae_metric.result().numpy())


@tf.function(experimental_relax_shapes=True)
def test_step(model, x_validation, y_validation, validation_metric):
    predictions = model(x_validation, training=False)
    mean_prediction = tf.reduce_mean(predictions, keepdims=True)
    validation_metric.update_state(y_validation, mean_prediction)
    return mean_prediction


def get_first_frames(video, number_of_frames):
    return video[:, :number_of_frames, :, :, :]


if __name__ == "__main__":
    # execute only if run as a script
    main()
