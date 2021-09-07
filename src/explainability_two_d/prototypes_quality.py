import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import explainability.read_helpers as rh
import shapely
from shapely.geometry import Polygon
from dtw import *
from pathlib import Path
from tensorflow import keras
from explainability import prototypes_quality_videos
from prototypes_calculation import get_images_of_prototypes
from two_D_resnet import get_data


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with still images.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save prototypes and evaluations in")
    parser.add_argument('-p', '--prototypes_filename', default='prototypes.txt',
                        help='Name of file containing prototypes')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-cc', '--volume_cluster_centers_file',
                        default='../../data/clustering_volume/cluster_centers_ef.txt',
                        help='Path to file containing volume cluster labels')
    parser.add_argument('-cb', '--volume_cluster_borders_file',
                        default='../../data/clustering_volume/cluster_upper_borders_ef.txt',
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
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)

    # get prototypes and corresponding segmentation coordinates
    volume_tracings_data_frame = pd.read_csv(frame_volumes_path)
    volume_tracings_data_frame = volume_tracings_data_frame.set_index('ImageFileName')
    volume_tracings_dict = volume_tracings_data_frame.to_dict(orient='index')
    prototypes = rh.read_prototypes(Path(output_directory, args.prototypes_filename), frame_volumes_path)
    prototypes = rh.get_segmentation_coordinates_of_prototypes(prototypes, volume_tracings_dict)
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            points = zip(prototypes[i][j].segmentation['X'],
                         prototypes[i][j].segmentation['Y'])
            # points = [list(x) for x in points]
            # points.sort(key=lambda x: math.atan2(x[1] - 0, x[0] - 0))
            poly = Polygon(points)
            if not poly.exterior.is_valid:
                print(prototypes[i][j].file_name)
                print(poly.is_valid)
                x, y = poly.exterior.xy
                plt.plot(x, y)
                for k in range(len(x)):
                    plt.annotate(k, (x[k], y[k]))
                plt.show()
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

    # get ef cluster centers
    volume_cluster_centers = rh.read_ef_cluster_centers(args.volume_cluster_centers_file)

    # get ef cluster borders
    volume_cluster_borders = rh.read_ef_cluster_centers(args.volume_cluster_borders_file)

    # validate clustering -> by validating prototypes
    evaluate_prototypes(
        volume_cluster_centers, volume_cluster_borders,
        prototypes, volume_tracings_dict,
        train_still_images, train_volumes, train_filenames,
        test_still_images, test_volumes, test_filenames,
        args.model_path, args.hidden_layer_index,
        args.output_directory)


def evaluate_prototypes(volume_cluster_centers, volume_cluster_borders,
                        prototypes, volume_tracings_dict,
                        train_still_images, train_volumes, train_filenames,
                        test_still_images, test_volumes, test_filenames,
                        model_path, hidden_layer_index,
                        output_directory):
    # iterate over testset/trainingset:
    #   (1) get its ef-cluster (by choosing the closest center)
    #   (2) compare each test video (or its extracted features) to all
    #   prototypes of this ef-cluster and return the most similar
    #   (3) calculate distance with given distance/similarity measure
    # save/evaluate distances

    # load model
    print('Start loading model')
    model = keras.models.load_model(model_path)
    print('End loading model')

    print('Number of layers', len(model.layers))
    if hidden_layer_index is None:
        hidden_layer_index = len(model.layers) - 2
    print('Hidden layer index', hidden_layer_index)
    predicting_model = keras.Model(inputs=[model.input],
                                   outputs=model.layers[len(model.layers) - 1].output)
    extractor = keras.Model(inputs=[model.input],
                            outputs=model.layers[hidden_layer_index].output)
    prototypes = get_images_of_prototypes(prototypes, train_still_images, train_filenames)

    prototype_still_images = []
    prototype_volumes = []
    prototype_filenames = []
    print("----------------")
    print("Number of EF clusters:", len(prototypes))
    for i in range(len(prototypes)):
        print("Number of video clusters in EF cluster", i, ":", len(prototypes[i]))
        for j in range(len(prototypes[i])):
            prototype_still_images.append(prototypes[i][j].video)
            prototype_volumes.append(prototypes[i][j].ef)
            prototype_filenames.append(prototypes[i][j].file_name)
    print(len(prototypes[0][0].features))

    similarity_measures = ['euclidean', 'cosine', 'ssim', 'psnr']

    calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        prototype_still_images, prototype_volumes, prototype_filenames,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, similarity_measures,
                        data='prototypes')
    return
    calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        test_still_images, test_volumes, test_filenames,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, similarity_measures,
                        data='test')
    calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        train_still_images, train_volumes, train_filenames,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, similarity_measures,
                        data='train')


def calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        still_images, volumes, file_names,
                        prototypes, volume_tracings_dict,
                        predicting_model, extractor,
                        output_directory, similarity_measures,
                        data='test'):
    ## diffs_volumes = {m: [] for m in similarity_measures}
    ## diffs_features = {m: [] for m in similarity_measures}
    ## diffs_images = {m: [] for m in similarity_measures}
    ## vol_prototype = {m: [] for m in similarity_measures}
    # save prototypes which are most similar
    chosen_prototypes = []

    for i in range(len(still_images)):
        print("")
        print("Instance: ", i)
        chosen_prototypes.append({})
        chosen_prototypes[i]['file_name'] = file_names[i]
        # get predicted volume
        instance = np.expand_dims(still_images[i], axis=0)
        prediction = float(predicting_model(instance).numpy()[0][0])
        print("Predicted volume:", prediction)
        chosen_prototypes[i]['predicted_volume'] = prediction

        # get actual volume label
        print("Actual volume:", volumes[i])
        chosen_prototypes[i]['actual_volume'] = volumes[i]
        chosen_prototypes[i]['prediction_error'] = abs(volumes[i] - prediction)

        # get volume cluster of video by choosing corresponding ef-range (i.e. use kde-borders)
        clustered = False
        volume_cluster_index = 0
        for j in range(len(volume_cluster_borders)):
            if prediction <= volume_cluster_borders[j]:
                volume_cluster_index = j
                clustered = True
                break
        if not clustered:
            volume_cluster_index = len(volume_cluster_borders) - 1
        print("Volume cluster index", volume_cluster_index)
        chosen_prototypes[i]['volume_cluster'] = volume_cluster_index

        # extract features
        extracted_features = extractor(instance)
        image = rh.Video(extracted_features, volumes[i], file_names[i], instance)

        # current_prototypes = None
        current_prototypes = []
        current_prototypes = prototypes[volume_cluster_index]
        if volume_cluster_index > 0:
            current_prototypes = current_prototypes + prototypes[volume_cluster_index - 1]
        if volume_cluster_index < len(prototypes) - 1:
            current_prototypes = current_prototypes + prototypes[volume_cluster_index + 1]
        print('Number considered prototypes', len(current_prototypes))

        # get most similar prototype of ef cluster
        # calculate distances/similarities

        euc_prototype, euc_index, euc_diff_features, euc_iou, euc_angle_diff, euc_volumes_diff, \
            cosine_prototype, cosine_index, cosine_diff_features, cosine_iou, cosine_angle_diff, cosine_volumes_diff = \
            prototypes_quality_videos.get_most_similar_prototypes(current_prototypes, image, volume_tracings_dict)

        # EUCLIDEAN DISTANCE
        chosen_prototypes[i]['euclidean_prototype'] = euc_prototype.file_name
        chosen_prototypes[i]['euclidean_volume'] = euc_prototype.ef
        chosen_prototypes[i]['euclidean_diff_volumes'] = euc_volumes_diff
        chosen_prototypes[i]['euclidean_diff_features'] = euc_diff_features
        chosen_prototypes[i]['euclidean_iou'] = euc_iou
        chosen_prototypes[i]['euclidean_diff_angles'] = euc_angle_diff
        print("Euclidean image cluster index", euc_index)
        print("Volume euclidean prototype: ", euc_prototype.ef)
        print("Euclidean diff features", euc_diff_features)
        print("Euclidean iou", euc_iou)
        print("Euclidean diff angles", euc_angle_diff)
        # euc_diff_images = np.linalg.norm([np.array(instance) - np.array(euc_prototype.video)])
        # print("Euclidean diff images:", euc_diff_images)
        # chosen_prototypes[i]['euclidean_diff_images'] = euc_diff_images

        # COSINE SIMILARITY (close to 1 indicates higher similarity)
        chosen_prototypes[i]['cosine_prototype'] = cosine_prototype.file_name
        chosen_prototypes[i]['cosine_volume'] = cosine_prototype.ef
        chosen_prototypes[i]['cosine_diff_volumes'] = cosine_volumes_diff
        chosen_prototypes[i]['cosine_diff_features'] = cosine_diff_features
        chosen_prototypes[i]['cosine_iou'] = cosine_iou
        chosen_prototypes[i]['cosine_diff_angles'] = cosine_angle_diff
        print("Cosine image cluster index", cosine_index)
        print("Volume cosine prototype: ", cosine_prototype.ef)
        print("Cosine diff features", cosine_diff_features)
        print("Cosine iou", cosine_iou)
        print("Cosine diff angles", cosine_angle_diff)
        # cosine_diff_images = np.linalg.norm(
        #     [np.array(instance) - np.array(cosine_prototype.video)])
        # print("Cosine diff images:", cosine_diff_images)
        # chosen_prototypes[i]['cosine_diff_images'] = cosine_diff_images

        continue
        # STRUCTURAL SIMILARITY (close to 1 indicates higher similarity)
        prototype, prototype_index = prototypes_quality_videos.get_most_similar_prototype_ssim(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['ssim_prototype'] = prototype.file_name
        chosen_prototypes[i]['ssim_volume'] = prototype.ef
        chosen_prototypes[i]['ssim_diff_volumes'] = abs(prototype.ef - volumes[i])
        diff_features = prototypes_quality_videos.structural_similarity(
            np.array(extracted_features[0]).astype('float64'), np.array(prototype.features),
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        chosen_prototypes[i]['ssim_diff_features'] = diff_features
        chosen_prototypes[i]['ssim_iou'] = 0
        chosen_prototypes[i]['ssim_diff_angles'] = 0
        print("SSIM image cluster index", prototype_index)
        print("SSIM cosine prototype: ", prototype.ef)
        print("SSIM diff features", diff_features)
        print("SSIM iou", "not calculated")
        print("SSIM diff angles", "not calculated")
        # SSIM_diff_images = prototypes_quality_videos.structural_similarity(
        #     np.array(np.array(instance[0])), np.array(prototype.video),
        #     gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        #     multichannel=True)
        # print("SSIM diff images:", SSIM_diff_images)
        # chosen_prototypes[i]['ssim_diff_images'] = SSIM_diff_images

        continue
        # PEAK-SIGNAL-TO-NOISE RATIO (higher is better)
        # TODO: [results seem wrong]
        prototype, prototype_index = prototypes_quality_videos.get_most_similar_prototype_psnr(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['psnr_prototype'] = prototype.file_name
        chosen_prototypes[i]['psnr_volume'] = prototype.ef
        chosen_prototypes[i]['psnr_diff_volumes'] = abs(prototype.ef - volumes[i])
        diff_features = prototypes_quality_videos.peak_signal_noise_ratio(
            np.array(extracted_features[0]), np.array(prototype.features),
            data_range=max(np.array(extracted_features[0])) - min(
                np.array(extracted_features[0])))
        chosen_prototypes[i]['psnr_diff_features'] = diff_features
        chosen_prototypes[i]['psnr_iou'] = 0
        chosen_prototypes[i]['psnr_diff_angles'] = 0
        print("PSNR image cluster index", prototype_index)
        print("PSNR cosine prototype: ", prototype.ef)
        print("PSNR diff features", diff_features)
        print("PSNR iou", "not calculated")
        print("PSNR diff angles", "not calculated")
        # PSNR_diff_images = prototypes_quality_videos.peak_signal_noise_ratio(
        #     np.array(instance).flatten(), np.array(prototype.video).flatten(),
        #     data_range=max(np.array(instance)) - min(np.array(instance)))
        # print("SSIM diff images:", SSIM_diff_images)
        # chosen_prototypes[i]['psnr_diff_images'] = SSIM_diff_images

    cp = pd.DataFrame(chosen_prototypes)
    prototypes_path = Path(output_directory, data + '_chosen_prototypes.csv')
    cp.to_csv(prototypes_path, index=False)


def compare_polygons_dtw(points_1, points_2):
    # using HMM Similarity Paper etc.
    # angles_1 = angles_for_points(points_1)
    # angles_2 = angles_for_points(points_2)
    alignment = dtw(points_1, points_2, keep_internals=True)
    # alignment.plot(type="threeway")
    print("Distance:", alignment.distance)
    return alignment.distance


def compare_angles(points_1, points_2):
    angles_1 = angles_for_points(points_1)
    angles_2 = angles_for_points(points_2)
    diff = 0
    for i in range(min(len(angles_1), len(angles_2))):
        diff = diff + abs(angles_1[i] - angles_2[i])
    return diff


def angles_for_points(points):
    def vector(tail, head):
        return tuple(h - t for h, t in zip(head, tail))

    points = points[:] + points[0:2]
    angles = []
    for p0, p1, p2 in zip(points, points[1:], points[2:]):
        v0 = vector(tail=p0, head=p1)
        a0 = math.atan2(v0[1], v0[0])
        v1 = vector(tail=p1, head=p2)
        a1 = math.atan2(v1[1], v1[0])
        angle = a1 - a0
        if angle < 0:
            angle += 2 * math.pi
        angles.append(angle)
    return angles


if __name__ == '__main__':
    main()
