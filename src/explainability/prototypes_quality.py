import argparse
import json
import numpy as np
import explainability.read_helpers as rh
from pathlib import Path
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from data_loader import tf_record_loader
from explainability.prototypes_calculation import get_videos_of_prototypes
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
                        output_directory, compare_videos=False):
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
    if compare_videos:
        prototypes = get_videos_of_prototypes(prototypes, metadata_filename, train_dataset, number_input_frames)

    similarity_measures = ['euclidean', 'cosine', 'ssim', 'psnr']

    calculate_distances(train_dataset, number_input_frames,
                        ef_cluster_centers, ef_cluster_borders,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        compare_videos, data='train')
    # calculate_distances(test_dataset, number_input_frames,
    #                     ef_cluster_centers, ef_cluster_borders,
    #                     prototypes, predicting_model, extractor,
    #                     output_directory, similarity_measures,
    #                     compare_videos, data='test')


def calculate_distances(dataset, number_input_frames, ef_cluster_centers,
                        ef_cluster_borders,
                        prototypes, predicting_model, extractor,
                        output_directory, similarity_measures,
                        compare_videos, data='train'):
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
        if compare_videos:
            save_distances(output_directory, data, sim, diffs_videos[sim], 'videos')

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


def get_most_similar_prototype_euclidean(prototypes, video, features=True):
    # compare given video to all prototypes of corresponding ef_cluster
    # return the one with the smallest distance
    # (here prototypes and video may also be given as extracted features)
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        min_dist = np.linalg.norm(
            [np.array(video.features) - np.array(most_similar.features)])
    else:
        min_dist = np.linalg.norm(
            [np.array(video.video) - np.array(most_similar.video)])
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_dist = np.linalg.norm(
                [np.array(video.features) - np.array(prototype.features)])
        else:
            current_dist = np.linalg.norm(
                [np.array(video.video) - np.array(prototype.video)])
        if current_dist < min_dist:
            most_similar = prototype
            most_similar_index = i
            min_dist = current_dist
        i += 1
    return most_similar, most_similar_index


def get_most_similar_prototype_cosine(prototypes, video, features=True):
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        max_sim = cosine_similarity(video.features, [most_similar.features])[0][0]
    else:
        max_sim = cosine_similarity(video.video, [most_similar.video])[0][0]
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_sim = cosine_similarity(video.features, [prototype.features])[0][0]
        else:
            current_sim = cosine_similarity(video.video, [prototype.video])[0][0]
        if current_sim > max_sim:
            most_similar = prototype
            most_similar_index = i
            max_sim = current_sim
        i += 1
    return most_similar, most_similar_index


def get_most_similar_prototype_ssim(prototypes, video, features=True):
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        max_sim = structural_similarity(
                np.array(video.features[0]).astype('float64'), np.array(most_similar.features),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    else:
        max_sim = structural_similarity(
                np.array(video.video[0]), np.array(most_similar.video),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_sim = structural_similarity(
                np.array(video.features[0]).astype('float64'), np.array(prototype.features),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        else:
            current_sim = structural_similarity(
                np.array(video.video[0]), np.array(prototype.video),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
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
