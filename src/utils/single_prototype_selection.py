import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow import keras
from utils import read_helpers as rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with still images.')
    parser.add_argument('-o', '--output_directory',
                        default='../../data/results',
                        help='Directory to save prototypes and evaluations in.')
    parser.add_argument('-f', '--file', default='0X1A2C60147AF9FDAE_62.png',
                        help='Image whose prototype should be calculated')
    parser.add_argument('-p', '--prototypes_filename', default='prototypes_esv.txt',
                        help='Name of file containing prototypes')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-cb', '--volume_cluster_borders_file',
                        default='../../data/clustering_volume/cluster_upper_borders_esv.txt',
                        help='Path to file containing volume cluster upper borders.')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-l', '--hidden_layer_index', default=86, type=int)
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'results')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # get volume cluster borders
    volume_cluster_borders = rh.read_volume_cluster_centers(args.volume_cluster_borders_file)

    # get prototypes
    prototypes = rh.read_prototypes(Path(output_directory, args.prototypes_filename))

    # validate clustering -> by validating prototypes
    calculate_prototype(
        volume_cluster_borders,
        prototypes,
        args.model_path, args.hidden_layer_index,
        args.input_directory, args.file)


def calculate_prototype(volume_cluster_borders,
                        prototypes,
                        model_path, hidden_layer_index,
                        input_directory, file):
    """Select the most similar prototype to the given still image file
    when for similarity measuring only the feature distance is considered."""
    # load model
    print('Start loading model')
    model = keras.models.load_model(model_path)
    print('End loading model')
    predicting_model = keras.Model(inputs=[model.input],
                                   outputs=model.layers[hidden_layer_index].output)

    extractor = keras.Model(inputs=[model.input],
                            outputs=model.layers[hidden_layer_index].output)

    image = Image.open(Path(input_directory, file))
    frame = np.asarray(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = frame / 255.0
    instance = np.expand_dims(frame, axis=0)
    prediction = float(predicting_model(instance).numpy()[0][0])
    print('Image:', file)
    print('Predicted Volume:', prediction)

    # get volume cluster of image by choosing corresponding volume-range
    clustered = False
    volume_cluster_index = 0
    for j in range(len(volume_cluster_borders)):
        if prediction <= volume_cluster_borders[j]:
            volume_cluster_index = j
            clustered = True
            break
    if not clustered:
        volume_cluster_index = len(volume_cluster_borders)
    print('Volume cluster index:', volume_cluster_index)
    current_prototypes = prototypes[volume_cluster_index]

    # extract features
    extracted_features = extractor(instance)

    # get most similar prototype of volume cluster
    # calculate distances/similarities
    euc_feature_diff = []
    i = 0
    for prototype in current_prototypes:
        # feature similarity using Euclidean distance
        euc_feature_diff.append(np.linalg.norm(
            [np.array(extracted_features[0]) - np.array(prototype.features)]))
        i += 1
    # get index of prototype with minimum difference
    most_similar_index = euc_feature_diff.index(min(euc_feature_diff))
    print('Most similar prototype:', current_prototypes[most_similar_index].file_name)
    print('Euclidean distance of features:', euc_feature_diff[most_similar_index])


if __name__ == '__main__':
    main()
