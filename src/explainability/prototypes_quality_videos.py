import argparse
import json
import ast
import math
import numpy as np
import explainability.read_helpers as rh
import explainability_two_d.prototypes_quality as pq
from shapely.geometry import Polygon
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from data_loader import tf_record_loader
from explainability.prototypes_calculation_videos import get_videos_of_prototypes
from explainability.clustering_ef import get_ef_cluster_centers_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the TFRecord files.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save prototypes and evaluations in")
    parser.add_argument('-p', '--prototypes_filename', default='prototypes.txt',
                        help='Name of file containing prototypes')
    parser.add_argument('-cc', '--ef_cluster_centers_file',
                        default='../../data/clustering_ef/cluster_centers_ef.txt',
                        help='Path to file containing ef cluster labels')
    parser.add_argument('-cb', '--ef_cluster_borders_file',
                        default='../../data/clustering_ef/cluster_upper_borders_ef.txt',
                        help='Path to file containing ef cluster upper borders')
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('-m', '--metadata_filename', default='FileList.csv',
                        help="Name of the metadata file.")
    parser.add_argument('-vc', '--video_clusters_directory',
                        default='../../data/video_clusters',
                        help='Directory with video cluster labels')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-l', '--hidden_layer_index', default=14, type=int)
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'results')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # load test dataset
    data_folder = Path(args.input_directory)
    test_record_file_names = data_folder / 'test' / 'test_*.tfrecord.gzip'
    test_dataset = tf_record_loader.build_dataset(
        file_names=str(test_record_file_names),
        batch_size=1,
        shuffle_size=None,
        number_of_input_frames=args.number_input_frames)

    # load train dataset
    train_record_file_names = data_folder / 'train' / 'train_*.tfrecord.gzip'
    validation_record_file_names = data_folder / 'validation' / 'validation_*.tfrecord.gzip'
    train_dataset = tf_record_loader.build_dataset(
        file_names=str(train_record_file_names),
        batch_size=1,
        shuffle_size=None,
        number_of_input_frames=args.number_input_frames)
    validation_dataset = tf_record_loader.build_dataset(
        file_names=str(validation_record_file_names),
        batch_size=1,
        shuffle_size=None,
        number_of_input_frames=args.number_input_frames)
    train_dataset = train_dataset.concatenate(validation_dataset)

    # get ef cluster centers
    ef_cluster_centers = rh.read_ef_cluster_centers(args.ef_cluster_centers_file)

    # get ef cluster borders
    ef_cluster_borders = rh.read_ef_cluster_centers(args.ef_cluster_borders_file)

    # get prototypes
    prototypes = rh.read_prototypes(
        Path(output_directory, args.prototypes_filename))

    # validate clustering -> by validating prototypes
    evaluate_prototypes(ef_cluster_centers, ef_cluster_borders,
                        prototypes, train_dataset,
                        test_dataset, args.metadata_filename, args.model_path,
                        args.hidden_layer_index, args.number_input_frames,
                        args.output_directory)


def evaluate_prototypes(ef_cluster_centers, ef_cluster_borders,
                        prototypes, train_dataset,
                        test_dataset, metadata_filename, model_path,
                        hidden_layer_index, number_input_frames,
                        output_directory):
    # iterate over testset/trainingset:
    #   (1) get its ef-cluster (by choosing the closest center)
    #   (2) compare each test video (or its extracted features) to all
    #   prototypes of this ef-cluster and return the most similar
    #   (3) calculate distance with given distance/similarity measure
    # save/evaluate distances

    # load model
    print("Start loading model")
    print(model_path)
    model = keras.models.load_model(model_path)
    print("End loading model")
    predicting_model = keras.Model(inputs=model.get_layer(index=0).input,
                                   outputs=model.layers[0].layers[hidden_layer_index + 1].output)
    extractor = keras.Model(inputs=model.get_layer(index=0).input,
                            outputs=model.layers[0].layers[hidden_layer_index].output)

    prototypes = get_videos_of_prototypes(prototypes, metadata_filename, train_dataset, number_input_frames)

    similarity_measures = ['euclidean', 'cosine', 'ssim', 'psnr']

    calculate_distances(train_dataset, number_input_frames,
                        ef_cluster_centers, ef_cluster_borders,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        data='train')
    # calculate_distances(test_dataset, number_input_frames,
    #                     ef_cluster_centers, ef_cluster_borders,
    #                     prototypes, predicting_model, extractor,
    #                     output_directory, similarity_measures,
    #                     data='test')


def calculate_distances(dataset, number_input_frames, ef_cluster_centers,
                        ef_cluster_borders,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        data='train'):
    diffs_ef = {m: [] for m in similarity_measures}
    diffs_features = {m: [] for m in similarity_measures}
    diffs_videos = {m: [] for m in similarity_measures}
    ef_y = []
    ef_predicted = []
    # save efs of prototype which is most similar
    ef_prototype = {m: [] for m in similarity_measures}
    prediction_error = []

    i = 0
    for video, y in dataset:
        print("Iteration: ", i)
        i += 1
        # get predicted ef
        first_frames = video[:, :number_input_frames, :, :, :]
        prediction = float(predicting_model(first_frames).numpy()[0][0])
        ef_predicted.append(prediction)

        # get actual ef label
        ef = float(y.numpy()[0])
        ef_y.append(ef)
        prediction_error.append(abs(ef - prediction))

        # get ef cluster of video by choosing the closest ef to predicted ef
        # ef_cluster_index = np.argmin([abs(prediction - e) for e in ef_cluster_centers])
        # alternative: choose corresponding ef-range (i.e. use kde-borders)
        clustered = False
        ef_cluster_index = 0
        for j in range(len(ef_cluster_borders)):
            if prediction <= ef_cluster_borders[j]:
                ef_cluster_index = j
                clustered = True
                break
        if not clustered:
            ef_cluster_index = len(ef_cluster_borders)

        # extract features
        flattened_video = first_frames.numpy().flatten()
        extracted_features = extractor(first_frames)
        video = rh.Video(extracted_features, None, None, flattened_video)

        # get most similar prototype of ef cluster
        # calculate distances/similarities

        # EUCLIDEAN DISTANCE
        prototype, prototype_index = get_most_similar_prototype_euclidean(
            prototypes[ef_cluster_index], video, features=True)
        ef_prototype[similarity_measures[0]].append(prototype.ef)
        diff_ef = abs(prototype.ef - ef)
        diffs_ef[similarity_measures[0]].append(diff_ef)
        diff_features = np.linalg.norm(
            [np.array(extracted_features) - np.array(prototype.features)])
        diffs_features[similarity_measures[0]].append(diff_features)
        # diff_videos = np.linalg.norm(
        #     [np.array(flattened_video) - np.array(prototype.video)])
        # diffs_videos[similarity_measures[0]].append(diff_videos)

        # COSINE SIMILARITY
        prototype, prototype_index = get_most_similar_prototype_cosine(
            prototypes[ef_cluster_index], video, features=True)
        # print("Considered file:", y)
        # print("File of closest prototype:", prototype.file_name)
        # print("EF prototype: ", prototype.ef)
        ef_prototype[similarity_measures[1]].append(prototype.ef)
        diff_ef = abs(prototype.ef - ef)
        diffs_ef[similarity_measures[1]].append(diff_ef)
        diff_features = cosine_similarity(extracted_features, [prototype.features])[0][0]
        diffs_features[similarity_measures[1]].append(diff_features)
        # diff_videos = cosine_similarity([flattened_video], [prototype.video])[0][0]
        # diffs_videos[similarity_measures[1]].append(diff_videos)

        # STRUCTURAL SIMILARITY
        prototype, prototype_index = get_most_similar_prototype_ssim(
            prototypes[ef_cluster_index], video, features=True)
        ef_prototype[similarity_measures[2]].append(prototype.ef)
        diff_ef = abs(prototype.ef - ef)
        diffs_ef[similarity_measures[2]].append(diff_ef)
        diff_features = structural_similarity(
            np.array(extracted_features[0]).astype('float64'), np.array(prototype.features),
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        diffs_features[similarity_measures[2]].append(diff_features)
        # diff_videos = structural_similarity(
        #     np.array(flattened_video), np.array(prototype.video),
        #     gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        # diffs_videos[similarity_measures[2]].append(diff_videos)

        # PEAK-SIGNAL-TO-NOISE RATIO
        prototype, prototype_index = get_most_similar_prototype_psnr(
            prototypes[ef_cluster_index], video, features=True)
        ef_prototype[similarity_measures[3]].append(prototype.ef)
        diff_ef = abs(prototype.ef - ef)
        diffs_ef[similarity_measures[3]].append(diff_ef)
        diff_features = peak_signal_noise_ratio(
            np.array(extracted_features[0]), np.array(prototype.features),
            data_range=max(np.array(extracted_features[0])) - min(
                np.array(extracted_features[0])))
        diffs_features[similarity_measures[3]].append(diff_features)
        # diff_videos = peak_signal_noise_ratio(
        #     np.array(flattened_video), np.array(prototype.video),
        #     data_range=max(np.array(flattened_video)) - min(
        #         np.array(flattened_video)))
        # diffs_videos[similarity_measures[3]].append(diff_videos)

    for sim in similarity_measures:
        save_distances(output_directory, data, sim, diffs_ef[sim], 'ef'
                                                                   '')
        save_distances(output_directory, data, sim, diffs_features[sim], 'features')
        # save_distances(output_directory, data, sim, diffs_videos[sim], 'videos')

    with open(Path(output_directory, data + '_prediction_error.txt'), 'w') as txt_file:
        for e in prediction_error:
            txt_file.write(str(e) + "\n")


def save_distances(output_directory, data, similarity_measure, diffs_data, diffs_name):
    with open(Path(output_directory, str(data) + '_diffs_' + diffs_name + '_' + similarity_measure + '.txt'), "w") as txt_file:
        for d in diffs_data:
            txt_file.write(str(d) + "\n")
    #save_metadata(diffs_data, Path(output_directory, str(data) + '_diffs_' + diffs_name + '_' + similarity_measure + '_metadata.json'))


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


def get_most_similar_prototypes(prototypes, video, volume_tracings_dict,
                                weights=[0.0, 0.0, 1.0, 0.0]): # [0.4, 0.2, 0.2, 0.2]
    # get polygon of current instance
    instance_polygon = Polygon(zip(
        ast.literal_eval(volume_tracings_dict[video.file_name]['X']),
        ast.literal_eval(volume_tracings_dict[video.file_name]['Y'])
    ))
    instance_points = [list(x) for x in zip(
        ast.literal_eval(volume_tracings_dict[video.file_name]['X']),
        ast.literal_eval(volume_tracings_dict[video.file_name]['Y'])
    )]
    # compare given video to all prototypes of corresponding ef_cluster
    # using multiple metrics
    # return the one with the smallest/minimum distance
    euc_feature_diff = []
    cosine_feature_diff = []
    iou = []
    angle_diff = []
    volumes_diff = []
    i = 0
    for prototype in prototypes:
        # feature similarity
        euc_feature_diff.append(np.linalg.norm([np.array(video.features) - np.array(prototype.features)]))
        cosine_feature_diff.append(-1 * (cosine_similarity(video.features, [prototype.features])[0][0]))

        # Intersection of Union (IoU) of left ventricle polygons
        prototype_polygon = Polygon(zip(
            prototypes[i].segmentation['X'],
            prototypes[i].segmentation['Y']
        ))
        prototype_points = [list(x) for x in zip(
            prototypes[i].segmentation['X'],
            prototypes[i].segmentation['Y']
        )]
        intersection_polygon = instance_polygon.intersection(prototype_polygon)
        intersection = intersection_polygon.area
        union = prototype_polygon.area + instance_polygon.area - intersection
        iou.append(-1 * (intersection / union))

        # angle similarity
        #angle_diff.append(pq.compare_polygons_dtw(instance_points, prototype_points))
        angle_diff.append(pq.compare_polygons_rotation_translation_invariant(prototype_points, instance_points))

        # volume similarity
        volumes_diff.append(abs(prototype.ef - video.ef))

        i += 1
    # standardize
    transformer = StandardScaler()
    euc_index = get_most_similar_prototype_index(euc_feature_diff, iou, angle_diff, volumes_diff, transformer, weights)
    cosine_index = get_most_similar_prototype_index(cosine_feature_diff, iou, angle_diff, volumes_diff, transformer, weights)
    return prototypes[euc_index], euc_index, euc_feature_diff[euc_index], -1 * iou[euc_index], angle_diff[euc_index], volumes_diff[euc_index], \
           prototypes[cosine_index], cosine_index, -1 * cosine_feature_diff[cosine_index], -1 * iou[cosine_index], angle_diff[cosine_index], volumes_diff[cosine_index]


def get_most_similar_prototype_index(feature_diff, iou, angle_diff, volumes_diff, transformer,
                                     weights):
    evaluation = [list(x) for x in zip(feature_diff, iou, angle_diff, volumes_diff)]
    # print('eval', evaluation)
    eval_ft = transformer.fit_transform(evaluation)
    # print('eval normed', eval_ft)
    weighted_sum = [weights[0] * x[0] + weights[1] * x[1] + weights[2] * x[2] + weights[3] * x[3] for x in eval_ft]
    # print('weighted sum', weighted_sum)
    # get index of prototype with minimum difference
    most_similar_index = weighted_sum.index(min(weighted_sum))
    return most_similar_index


def get_most_similar_prototype_ssim(prototypes, video, features=True):
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        max_sim = structural_similarity(
                np.array(video.features[0]).astype('float64'), np.array(most_similar.features),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    else:
        print(video.video.shape)
        print(most_similar.video.shape)
        max_sim = structural_similarity(
                np.array(video.video[0]), np.array(most_similar.video),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                multichannel=True)
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_sim = structural_similarity(
                np.array(video.features[0]).astype('float64'), np.array(prototype.features),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        else:
            current_sim = structural_similarity(
                np.array(video.video[0]), np.array(prototype.video),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                multichannel=True)
        if current_sim > max_sim:
            most_similar = prototype
            most_similar_index = i
            max_sim = current_sim
        i += 1
    return most_similar, most_similar_index


def get_most_similar_prototype_psnr(prototypes, video, features=True):
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        max_sim = peak_signal_noise_ratio(
            np.array(video.features[0]), np.array(most_similar.features),
            data_range=max(np.array(video.features[0]) - min(np.array(video.features[0])))
        )
    else:
        max_sim = peak_signal_noise_ratio(
            np.array(video.video[0]), np.array(most_similar.video),
            data_range=max(np.array(video.video[0])) - min(np.array(video.video[0]))
        )
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_sim = peak_signal_noise_ratio(
                np.array(video.features[0]), np.array(prototype.features),
                data_range=max(np.array(video.features[0])) - min(np.array(video.features[0]))
            )
        else:
            current_sim = peak_signal_noise_ratio(
                np.array(video.video[0]), np.array(prototype.video),
                data_range=max(np.array(video.video[0])) - min(np.array(video.video[0]))
            )
        if current_sim > max_sim:
            most_similar = prototype
            most_similar_index = i
            max_sim = current_sim
        i += 1
    return most_similar, most_similar_index


if __name__ == '__main__':
    main()
