from __future__ import division
import argparse
import numpy as np
import pandas as pd
import read_helpers as rh
from pathlib import Path
from tensorflow import keras
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from prototypes_calculation import get_images_of_prototypes
from two_d_resnet import get_data
import similarity as sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with still images.')
    parser.add_argument('-o', '--output_directory',
                        help='Directory to save prototypes and evaluations in.')
    parser.add_argument('-p', '--prototypes_filename', default='prototypes.txt',
                        help='Name of file containing prototypes')
    parser.add_argument('-re', '--rotation_extent', type=float, default=np.pi/8)
    parser.add_argument('-nr', '--num_rotations', type=int, default=9)
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-cc', '--volume_cluster_centers_file',
                        default='../../data/clustering_volume/cluster_centers_esv.txt',
                        help='Path to file containing volume cluster labels')
    parser.add_argument('-cb', '--volume_cluster_borders_file',
                        default='../../data/clustering_volume/cluster_upper_borders_esv.txt',
                        help='Path to file containing volume cluster upper borders')
    parser.add_argument('-if', '--image_features_directory',
                        default='../../data/image_features',
                        help='Directory with image features')
    parser.add_argument('-ic', '--image_clusters_directory',
                        default='../../data/image_clusters',
                        help='Directory with image cluster labels')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-l', '--hidden_layer_index', type=int)
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'image_clusters')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)

    # get prototypes and corresponding segmentation coordinates
    volume_tracings_data_frame = pd.read_csv(frame_volumes_path)
    volume_tracings_data_frame = volume_tracings_data_frame.set_index('ImageFileName')
    volume_tracings_dict = volume_tracings_data_frame.to_dict(orient='index')
    prototypes = rh.read_prototypes(Path(output_directory, args.prototypes_filename), frame_volumes_path)
    prototypes = rh.get_segmentation_coordinates_of_prototypes(prototypes, volume_tracings_dict)
    prototypes = rh.get_normalized_rotations_of_prototypes_with_angles(
        prototypes, args.rotation_extent, args.num_rotations)
    print('Prototype polygons saved')

    # get train/validation/test data
    train_still_images, train_volumes, train_filenames, \
        test_still_images, test_volumes, test_filenames, \
        val_still_images, val_volumes, val_filenames = get_data(
            args.input_directory, frame_volumes_path, volume_type=args.volume_type)
    train_still_images.extend(val_still_images)
    train_volumes.extend(val_volumes)
    train_filenames.extend(val_filenames)
    print('Data loaded')

    # get volume cluster centers
    # volume_cluster_centers = rh.read_volume_cluster_centers(args.volume_cluster_centers_file)

    # get volume cluster borders
    volume_cluster_borders = rh.read_volume_cluster_centers(args.volume_cluster_borders_file)

    # validate clustering -> by validating prototypes
    evaluate_prototypes(
        volume_cluster_borders,
        prototypes, volume_tracings_dict,
        train_still_images, train_volumes, train_filenames,
        test_still_images, test_volumes, test_filenames,
        args.model_path, args.hidden_layer_index,
        args.output_directory)


def evaluate_prototypes(volume_cluster_borders,
                        prototypes, volume_tracings_dict,
                        train_still_images, train_volumes, train_filenames,
                        test_still_images, test_volumes, test_filenames,
                        model_path, hidden_layer_index,
                        output_directory):
    # iterate over testset/trainingset:
    #   (1) get for current instance the corresponding volume-cluster
    #   by choosing the closest center
    #   (2) compare the extracted features of current test image to all
    #   prototypes of the volume cluster selected in (1) and
    #   return the most similar
    #   (3) calculate distance to most similar prototype selected in (2) with
    #   given distance/similarity measure
    # save/evaluate distances

    # load model
    print('Start loading model')
    model = keras.models.load_model(model_path)
    print('End loading model')

    if hidden_layer_index is None:
        hidden_layer_index = len(model.layers) - 2
    predicting_model = keras.Model(inputs=[model.input],
                                   outputs=model.layers[len(model.layers) - 1].output)
    extractor = keras.Model(inputs=[model.input],
                            outputs=model.layers[hidden_layer_index].output)
    prototypes = get_images_of_prototypes(prototypes, train_still_images, train_filenames)

    prototype_still_images = []
    prototype_volumes = []
    prototype_filenames = []
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            prototype_still_images.append(prototypes[i][j].image)
            prototype_volumes.append(prototypes[i][j].volume)
            prototype_filenames.append(prototypes[i][j].file_name)

    calculate_distances(volume_cluster_borders,
                        prototype_still_images, prototype_volumes, prototype_filenames,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, data='prototypes')
    calculate_distances(volume_cluster_borders,
                        test_still_images, test_volumes, test_filenames,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, data='test')
    calculate_distances(volume_cluster_borders,
                        train_still_images, train_volumes, train_filenames,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, data='train')


def calculate_distances(volume_cluster_borders,
                        still_images, volumes, file_names,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, data='test'):
    # save prototypes which are most similar
    chosen_prototypes = []

    for i in range(len(still_images)):
        chosen_prototypes.append({})
        chosen_prototypes[i]['file_name'] = file_names[i]
        # get predicted volume
        instance = np.expand_dims(still_images[i], axis=0)
        prediction = float(predicting_model(instance).numpy()[0][0])
        chosen_prototypes[i]['predicted_volume'] = prediction

        # get actual volume label
        chosen_prototypes[i]['actual_volume'] = volumes[i]
        chosen_prototypes[i]['prediction_error'] = abs(volumes[i] - prediction)

        # get volume cluster of image by choosing corresponding volume range
        clustered = False
        volume_cluster_index = 0
        for j in range(len(volume_cluster_borders)):
            if prediction <= volume_cluster_borders[j]:
                volume_cluster_index = j
                clustered = True
                break
        if not clustered:
            volume_cluster_index = len(volume_cluster_borders) - 1
        chosen_prototypes[i]['volume_cluster'] = volume_cluster_index

        # extract features
        extracted_features = extractor(instance)
        image = rh.Image(extracted_features, volumes[i], file_names[i], instance)

        # current_prototypes = None
        current_prototypes = []
        current_prototypes = prototypes[volume_cluster_index]
        if volume_cluster_index > 0:
            current_prototypes = current_prototypes + prototypes[volume_cluster_index - 1]
        if volume_cluster_index < len(prototypes) - 1:
            current_prototypes = current_prototypes + prototypes[volume_cluster_index + 1]

        # get most similar prototype of volume cluster
        # calculate distances/similarities regarding euclidean distance
        # and cosine similarity
        euc_prototype, euc_index, euc_diff_features, euc_iou, euc_angle_diff, \
            cosine_prototype, cosine_index, cosine_diff_features, cosine_iou, cosine_angle_diff = \
            sim.get_most_similar_prototype(current_prototypes, image, volume_tracings_dict)

        # EUCLIDEAN DISTANCE
        chosen_prototypes[i]['euclidean_prototype'] = euc_prototype.file_name
        chosen_prototypes[i]['euclidean_volume'] = euc_prototype.volume
        chosen_prototypes[i]['euclidean_diff_volumes'] = abs(euc_prototype.volume - volumes[i])
        chosen_prototypes[i]['euclidean_diff_features'] = euc_diff_features
        chosen_prototypes[i]['euclidean_iou'] = euc_iou
        chosen_prototypes[i]['euclidean_diff_angles'] = euc_angle_diff

        # COSINE SIMILARITY (close to 1 indicates higher similarity)
        chosen_prototypes[i]['cosine_prototype'] = cosine_prototype.file_name
        chosen_prototypes[i]['cosine_volume'] = cosine_prototype.volume
        chosen_prototypes[i]['cosine_diff_volumes'] = abs(cosine_prototype.volume - volumes[i])
        chosen_prototypes[i]['cosine_diff_features'] = cosine_diff_features
        chosen_prototypes[i]['cosine_iou'] = cosine_iou
        chosen_prototypes[i]['cosine_diff_angles'] = cosine_angle_diff

        continue
        # STRUCTURAL SIMILARITY (close to 1 indicates higher similarity)
        prototype, prototype_index = get_most_similar_prototype_ssim(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['ssim_prototype'] = prototype.file_name
        chosen_prototypes[i]['ssim_volume'] = prototype.volume
        chosen_prototypes[i]['ssim_diff_volumes'] = abs(prototype.volume - volumes[i])
        diff_features = structural_similarity(
            np.array(extracted_features[0]).astype('float64'), np.array(prototype.features),
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        chosen_prototypes[i]['ssim_diff_features'] = diff_features
        chosen_prototypes[i]['ssim_iou'] = 0
        chosen_prototypes[i]['ssim_diff_angles'] = 0

        # PEAK-SIGNAL-TO-NOISE RATIO (higher is better)
        prototype, prototype_index = get_most_similar_prototype_psnr(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['psnr_prototype'] = prototype.file_name
        chosen_prototypes[i]['psnr_volume'] = prototype.volume
        chosen_prototypes[i]['psnr_diff_volumes'] = abs(prototype.volume - volumes[i])
        diff_features = peak_signal_noise_ratio(
            np.array(extracted_features[0]), np.array(prototype.features),
            data_range=max(np.array(extracted_features[0])) - min(
                np.array(extracted_features[0])))
        chosen_prototypes[i]['psnr_diff_features'] = diff_features
        chosen_prototypes[i]['psnr_iou'] = 0
        chosen_prototypes[i]['psnr_diff_angles'] = 0

    print('Closest prototypes selected for all', data, 'instances')
    cp = pd.DataFrame(chosen_prototypes)
    prototypes_path = Path(output_directory, data + '_chosen_prototypes.csv')
    cp.to_csv(prototypes_path, index=False)


def save_metadata(diffs, output_file):
    metadata = {
        'max': str(max(diffs)),
        'min': str(min(diffs)),
        'sum': str(sum(diffs)),
        'mean': str(np.mean(diffs)),
        'std': str(np.std(diffs))
    }
    metadata = {'metadata': metadata}

    with open(output_file, 'w') as json_file:
        json.dump(metadata, json_file)


if __name__ == '__main__':
    main()
