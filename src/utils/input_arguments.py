import argparse


def get_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../data', help="Directory with the TFRecord files.")
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-s', '--shuffle_size', default=1024, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-p', '--patience', default=10, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('--dataset', default='stanford', choices=['stanford', 'mainz'])
    parser.add_argument('-t', '--target', default='ejection_fraction', choices=['ejection_fraction', 'e_e_prime', 'gls',
                                                                                'quality', 'all'])
    parser.add_argument('-m', '--model_name', default='resnet_18', choices=['vgg', 'resnet_18', 'resnet_34',
                                                                            'resnet_50', 'se-resnet_18',
                                                                            'se-resnet_34'])
    parser.add_argument('-n', '--experiment_name', required=True)
    parser.add_argument('--resolution', nargs='+', metavar=('height', 'width'), type=int)

    parser.add_argument('--augmentation', dest='augmentation', action='store_true')
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    parser.set_defaults(augmentation=True)

    parser.add_argument('--regularization', default='l2', choices=['l1', 'l2', 'l1_l2'])

    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true')
    parser.set_defaults(load_checkpoint=False)

    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.set_defaults(cache=False)

    return parser.parse_args()


def get_test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-i', '--input_directory', default='../data', help="Directory with the TFRecord files.")
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('--dataset', default='stanford', choices=['stanford', 'mainz'])
    parser.add_argument('-t', '--target', default='ejection_fraction', choices=['ejection_fraction', 'e_e_prime', 'gls',
                                                                                'quality'])
    parser.add_argument('--resolution', nargs='+', metavar=('height', 'width'), type=int)
    parser.add_argument('--model_path', required=True)
    return parser.parse_args()
