import argparse
import os
import numpy as np
import pandas as pd
import explainability.read_helpers as rh
from pathlib import Path
from tensorflow import keras
from explainability import prototypes_quality_videos
from two_D_resnet import get_data
from prototypes_calculation import get_images_of_prototypes


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

    # get prototypes
    prototypes = rh.read_prototypes(Path(output_directory, args.prototypes_filename))

    # validate clustering -> by validating prototypes
    evaluate_prototypes(
        volume_cluster_centers, volume_cluster_borders,
        prototypes,
        train_still_images, train_volumes, train_filenames,
        test_still_images, test_volumes, test_filenames,
        args.model_path, args.hidden_layer_index,
        args.output_directory)


def evaluate_prototypes(volume_cluster_centers, volume_cluster_borders,
                        prototypes,
                        train_still_images, train_volumes, train_filenames,
                        test_still_images, test_volumes, test_filenames,
                        model_path, hidden_layer_index,
                        output_directory, compare_images=True,
                        use_features=True):
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
    if compare_images:
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
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        compare_images, data='prototypes', use_features=use_features)
    calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        test_still_images, test_volumes, test_filenames,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        compare_images, data='test', use_features=use_features)
    calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        train_still_images, train_volumes, train_filenames,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        compare_images, data='train', use_features=use_features)


def calculate_distances(volume_cluster_centers, volume_cluster_borders,
                        still_images, volumes, file_names,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        compare_images, data='test', use_features=True):
    diffs_volumes = {m: [] for m in similarity_measures}
    diffs_features = {m: [] for m in similarity_measures}
    diffs_images = {m: [] for m in similarity_measures}
    vol_y = []
    vol_predicted = []
    # save volumes of prototype which is most similar
    vol_prototype = {m: [] for m in similarity_measures}
    prediction_error = []
    chosen_prototypes = []

    for i in range(len(still_images)):
        print("")
        print("Instance: ", i)
        chosen_prototypes.append({})
        # get predicted volume
        instance = np.expand_dims(still_images[i], axis=0)
        prediction = float(predicting_model(instance).numpy()[0][0])
        vol_predicted.append(prediction)
        print("Predicted volume:", prediction)
        chosen_prototypes[i]['predicted_volume'] = prediction

        # get actual volume label
        vol_y.append(volumes[i])
        print("Actual volume:", volumes[i])
        chosen_prototypes[i]['actual_volume'] = volumes[i]
        prediction_error.append(abs(volumes[i] - prediction))

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
        image = rh.Video(extracted_features, None, None, instance)

        # current_prototypes = None
        current_prototypes = []
        print('Number considered prototypes', len(current_prototypes))
        current_prototypes = prototypes[volume_cluster_index]
        if volume_cluster_index > 0:
            current_prototypes = current_prototypes + prototypes[volume_cluster_index - 1]
        if volume_cluster_index < len(prototypes) - 1:
            current_prototypes = current_prototypes + prototypes[volume_cluster_index + 1]
        print('Number considered prototypes', len(current_prototypes))

        # get most similar prototype of ef cluster
        # calculate distances/similarities

        # EUCLIDEAN DISTANCE
        prototype, prototype_index = prototypes_quality_videos.get_most_similar_prototype_euclidean(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['file_name'] = file_names[i]
        chosen_prototypes[i]['euclidean_prototype'] = prototype.file_name
        print("Euclidean image cluster index", prototype_index)
        print("Volume prototype: ", prototype.ef)
        chosen_prototypes[i]['euclidean_volume'] = prototype.ef
        vol_prototype[similarity_measures[0]].append(prototype.ef)
        diff_volume = abs(prototype.ef - volumes[i])
        diffs_volumes[similarity_measures[0]].append(diff_volume)
        diff_features = np.linalg.norm(
            [np.array(extracted_features) - np.array(prototype.features)])
        diffs_features[similarity_measures[0]].append(diff_features)
        chosen_prototypes[i]['euclidean_features'] = diff_features
        if compare_images:
            diff_images = np.linalg.norm([np.array(instance) - np.array(prototype.video)])
            diffs_images[similarity_measures[0]].append(diff_images)
            print("Sim:", diff_images)
            chosen_prototypes[i]['euclidean_images'] = diff_images

        # COSINE SIMILARITY (close to 1 indicates higher similarity)
        prototype, prototype_index = prototypes_quality_videos.get_most_similar_prototype_cosine(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['cosine_prototype'] = prototype.file_name
        print("Cosine image cluster index", prototype_index)
        print("Volume prototype: ", prototype.ef)
        chosen_prototypes[i]['cosine_volume'] = prototype.ef
        vol_prototype[similarity_measures[1]].append(prototype.ef)
        diff_volume = abs(prototype.ef - volumes[i])
        diffs_volumes[similarity_measures[1]].append(diff_volume)
        diff_features = prototypes_quality_videos.cosine_similarity(extracted_features, [prototype.features])[0][0]
        diffs_features[similarity_measures[1]].append(diff_features)
        chosen_prototypes[i]['cosine_features'] = diff_features
        if compare_images:
            diff_images = prototypes_quality_videos.cosine_similarity([np.array(instance).flatten()], [np.array(prototype.video).flatten()])[0][0]
            diffs_images[similarity_measures[1]].append(diff_images)
            print("Sim:", diff_images)
            chosen_prototypes[i]['cosine_images'] = diff_images

        continue
        # STRUCTURAL SIMILARITY (close to 1 indicates higher similarity)
        prototype, prototype_index = prototypes_quality_videos.get_most_similar_prototype_ssim(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['ssim_prototype'] = prototype.file_name
        print("SSIM image cluster index", prototype_index)
        print("Volume prototype: ", prototype.ef)
        chosen_prototypes[i]['ssim_volume'] = prototype.ef
        vol_prototype[similarity_measures[2]].append(prototype.ef)
        diff_volume = abs(prototype.ef - volumes[i])
        diffs_volumes[similarity_measures[2]].append(diff_volume)
        diff_features = prototypes_quality_videos.structural_similarity(
            np.array(extracted_features[0]).astype('float64'), np.array(prototype.features),
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        diffs_features[similarity_measures[2]].append(diff_features)
        chosen_prototypes[i]['ssim_features'] = diff_features
        if compare_images:
            print(instance.shape)
            print(prototype.video.shape)
            diff_images = prototypes_quality_videos.structural_similarity(
                np.array(np.array(instance[0])), np.array(prototype.video),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                multichannel=True)
            diffs_images[similarity_measures[2]].append(diff_images)
            print("Sim:", diff_images)
            chosen_prototypes[i]['ssim_images'] = diff_images

        continue
        # PEAK-SIGNAL-TO-NOISE RATIO (higher is better)
        # TODO: [results seem wrong]
        prototype, prototype_index = prototypes_quality_videos.get_most_similar_prototype_psnr(
            current_prototypes, image, features=use_features)
        chosen_prototypes[i]['psnr_prototype'] = prototype.file_name
        vol_prototype[similarity_measures[3]].append(prototype.ef)
        diff_volume = abs(prototype.ef - volumes[i])
        diffs_volumes[similarity_measures[3]].append(diff_volume)
        diff_features = prototypes_quality_videos.peak_signal_noise_ratio(
            np.array(extracted_features[0]), np.array(prototype.features),
            data_range=max(np.array(extracted_features[0])) - min(
                np.array(extracted_features[0])))
        diffs_features[similarity_measures[3]].append(diff_features)
        if compare_images:
            diff_images = prototypes_quality_videos.peak_signal_noise_ratio(
                np.array(instance).flatten(), np.array(prototype.video).flatten(),
                data_range=max(np.array(instance)) - min(np.array(instance)))
            diffs_images[similarity_measures[3]].append(diff_images)

    for sim in similarity_measures:
        prototypes_quality_videos.save_distances(output_directory, data, sim, diffs_volumes[sim], 'volume')
        prototypes_quality_videos.save_distances(output_directory, data, sim, diffs_features[sim], 'features')
        if compare_images:
            prototypes_quality_videos.save_distances(output_directory, data, sim, diffs_images[sim], 'images')
    cp = pd.DataFrame(chosen_prototypes)
    cp.to_csv('data/chosen_prototypes.csv')

    with open(Path(output_directory, data + '_prediction_error.txt'), 'w') as txt_file:
        for e in prediction_error:
            txt_file.write(str(e) + "\n")


if __name__ == '__main__':
    main()
