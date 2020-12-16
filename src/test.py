import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from data_loader import stanford_recordloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('--dataset', default='stanford', choices=['stanford', 'mainz'])
    parser.add_argument('-m', '--model_path', default='../../saved/three_d_conv_best_model.hd5',
                        help="Directory with the trained model.")
    parser.add_argument('-d', '--dataset_path', default='../../data/tf_record',
                        help="Directory with the TFRecord files.")
    args = parser.parse_args()
    test(args.batch_size, args.number_input_frames, args.dataset, args.model_path, args.dataset_path)


def test(batch_size, number_input_frames, dataset, model_path, dataset_path):
    data_folder = Path(dataset_path, 'tf_record')
    test_record_file_name = data_folder / 'test' / 'test_*.tfrecord'

    test_set = stanford_recordloader.build_dataset_validation(str(test_record_file_name), batch_size)
    model = keras.models.load_model(model_path)

    print("Test MSE:", results)


@tf.function
def test_step(model, loss_fn, x_batch_validation, y_batch_validation, test_mse_metric):
    predictions = model(x_batch_validation, training=True)
    loss_value = loss_fn(y_batch_validation, predictions)
    test_mse_metric.update_state(loss_value)


if __name__ == "__main__":
    # execute only if run as a script
    test()
