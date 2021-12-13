import argparse
import numpy as np
from pathlib import Path
from tensorflow import keras
from model.two_d_resnet import get_data
from utils import read_helpers as rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with still images.')
    parser.add_argument('-o', '--output_directory',
                        help='Directory to save the image features in.')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-cl', '--volume_clusters_file',
                        default='../../data/clustering_volume/cluster_labels_esv.txt',
                        help='Path to file containing volume cluster labels')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-l', '--hidden_layer_index', type=int)
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'image_features')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)

    # get data of volume clustering
    volume_cluster_labels, actual_volumes, file_names = rh.read_cluster_labels(args.volume_clusters_file)

    # get train/validation/test data
    train_still_images, _, _, _, _, _, val_still_images, _, _ = get_data(
        args.input_directory, frame_volumes_path, volume_type=args.volume_type)
    train_still_images.extend(val_still_images)
    print('Data loaded')

    extract_features(args.model_path, args.hidden_layer_index,
                     volume_cluster_labels, train_still_images, output_directory)


def extract_features(model_path, hidden_layer_index, volume_cluster_labels,
                     train_dataset, output_directory):
    """
    Extract features of all still images of each volume cluster at the given
    hidden layer index. The latent features are saved to files.
    @param model_path: model to use for latent feature extraction
    @param hidden_layer_index:  index of layer of model from which the features
    should be extracted
    @param volume_cluster_labels: cluster labels of volume clustering done
    before
    @param train_dataset: dataset consisting of training and validation data
    @param output_directory: directory for saving the extracted features
    """
    # load model
    print('Start loading model')
    model = keras.models.load_model(model_path)
    print('End loading model')

    # extract features
    if hidden_layer_index is None:
        hidden_layer_index = len(model.layers) - 2
    num_volume_clusters = max(volume_cluster_labels) + 1
    for i in range(num_volume_clusters):
        print('Extract features for volume cluster', i)
        extracted_features = get_hidden_layer_features(
            model, train_dataset, volume_cluster_labels, i,
            hidden_layer_index)

        # write features to file
        out_file = Path(output_directory, 'extracted_image_features_' + str(i) + '.txt')
        with open(out_file, 'w') as txt_file:
            for f in range(len(extracted_features)):
                txt_file.write(str(extracted_features[f]) + '\n')


def get_hidden_layer_features(model, train_dataset, cluster_labels,
                              volume_cluster_index, hidden_layer_index):
    """
    Extract features at given layer_index of given model for the given
    volume cluster.
    @return: list of features
    """
    extractor = keras.Model(inputs=[model.input],
                            outputs=model.layers[hidden_layer_index].output)
    extracted_features = []
    i = 0
    for instance in train_dataset:
        if cluster_labels[i] == volume_cluster_index:
            instance = np.expand_dims(instance, axis=0)
            features = extractor(instance)
            extracted_features.append(features)
        i += 1
    return extracted_features


if __name__ == '__main__':
    main()
