import argparse


def get_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../data', help="Directory with the TFRecord files.")
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-s', '--shuffle_size', default=1024, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-p', '--patience', default=10, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('--dataset', default='stanford', choices=['stanford', 'mainz'])
    parser.add_argument('-m', '--model_name', default='vgg', choices=['vgg', 'resnet_18', 'resnet_34',
                                                                      'resnet_50', 'se-resnet_18'])
    parser.add_argument('-a', '--augment', default=True, type=bool)
    parser.add_argument('-n', '--experiment_name', required=True)
    return parser.parse_args()
