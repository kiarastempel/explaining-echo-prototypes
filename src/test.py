import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import utils.input_arguments
from models.three_D_vgg_net import ThreeDConvolutionVGG
from models.three_D_resnet import ThreeDConvolutionResNet18, ThreeDConvolutionResNet34, ThreeDConvolutionResNet50
from models.three_D_squeeze_and_excitation_resnet import ThreeDConvolutionSqueezeAndExciationResNet18

from data_loader import stanford_recordloader, mainz_recordloader


def main():
    args = utils.input_arguments.get_test_arguments()
    test(args.batch_size, args.number_input_frames, args.dataset, args.model_weights_path, args.input_directory,
         args.model_name)


def test(batch_size, number_input_frames, dataset, model_weights_path, input_directory, model_name):
    data_folder = Path(input_directory)
    test_record_file_name = data_folder / 'test' / 'test_*.tfrecord.gzip'

    if model_name == 'resnet_18':
        model = ThreeDConvolutionResNet18(width, height, None, None, None, None)
    elif model_name == 'resnet_34':
        model = ThreeDConvolutionResNet34(width, height, number_input_frames, channels, mean, std)
    elif model_name == 'resnet_50':
        model = ThreeDConvolutionResNet50(width, height, number_input_frames, channels, mean, std)
    elif model_name == 'se-resnet_18':
        model = ThreeDConvolutionSqueezeAndExciationResNet18(width, height, number_input_frames, channels, mean, std)
    else:
        model = ThreeDConvolutionVGG(width, height, number_input_frames, channels, mean, std)

    test_set = stanford_recordloader.build_dataset_validation(str(test_record_file_name))
    model = keras.models.load_model(model_weights_path)

    print("Test MSE:", results)


@tf.function
def test_step(model, loss_fn, x_batch_validation, y_batch_validation, test_mse_metric):
    predictions = model(x_batch_validation, training=True)
    loss_value = loss_fn(y_batch_validation, predictions)
    test_mse_metric.update_state(loss_value)


if __name__ == "__main__":
    # execute only if run as a script
    main()
