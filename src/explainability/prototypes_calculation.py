import argparse
import pandas
import numpy as np
import os
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
import read_helpers as rh
from data_loader import tf_record_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the TFRecord files.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save prototypes and evaluations in")
    parser.add_argument('-p', '--prototypes_filename', default='prototypes.txt',
                        help='Name of file to save prototypes in')
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('-m', '--metadata_filename', default='FileList.csv',
                        help="Name of the metadata file.")
    parser.add_argument('-vf', '--video_features_directory',
                        default='../../data/video_features',
                        help='Directory with video features')
    parser.add_argument('-vc', '--video_clusters_directory',
                        default='../../data/video_clusters',
                        help='Directory with video cluster labels')
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    if output_directory is None:
        output_directory = Path(args.input_directory, 'results')
    output_directory.mkdir(parents=True, exist_ok=True)

    # load train/validation dataset
    data_folder = Path(args.input_directory)
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

    calculate_prototypes(args.video_features_directory,
                         args.video_clusters_directory,
                         Path(output_directory, args.prototypes_filename),
                         train_dataset, args.metadata_filename,
                         args.number_input_frames,
                         method='kmedoids', similarity_measure='euclidean')


def calculate_prototypes(video_features_directory, video_clusters_directory,
                         output_file, train_dataset, metadata_filename,
                         number_input_frames, method,
                         similarity_measure='euclidean'):
    methods = ['kmedoids', 'kmeans_features', 'kmeans_videos', 'cipa_features']
    if method not in methods:
        raise ValueError("ERROR: method has to be in", methods)
    num_ef_clusters = int(len(os.listdir(video_clusters_directory)) / 2)

    # store prototypes in map: ef(-index) -> [prototypes]
    # (one prototype per video cluster)
    prototypes = {}
    if method == 'kmedoids':
        prototypes = get_kmedoids_prototype_videos(num_ef_clusters,
                                                   video_clusters_directory,
                                                   metadata_filename,
                                                   train_dataset,
                                                   number_input_frames)
    elif method == 'kmeans_features' or method == 'cipa_features':
        prototypes = get_kmeans_prototype_features(num_ef_clusters,
                                                   video_clusters_directory)

    elif method == 'kmeans_videos':
        prototypes = get_kmeans_prototype_videos(num_ef_clusters,
                                                 video_features_directory,
                                                 video_clusters_directory,
                                                 metadata_filename,
                                                 train_dataset,
                                                 number_input_frames,
                                                 similarity_measure)

    print("Prototypes calculated")
    with open(output_file, "w") as txt_file:
        # TODO: change here depending on prototypes structure (video/features)
        for cp in prototypes:
            for i, p in enumerate(prototypes[cp]):
                txt_file.write(str(cp) + " "
                               + str(i) + " "
                               + str(p.ef) + " "
                               + str(p.file_name) + " "
                               + str(np.array(p.features)) + "\n")
                               # + str(p.video) + "\n")
    return prototypes


def get_kmedoids_prototype_videos(num_ef_clusters, video_clusters_directory,
                                  metadata_filename, train_dataset,
                                  number_input_frames):
    """
    Calculate prototypes of video clustering created by kmediods, where a
    prototype is defined as the medoid of a video cluster (i.e. it is a video).
    @param num_ef_clusters: number of clusters produced while clustering by EF
    @param video_clusters_directory: directory containing video cluster labels
    @param metadata_filename: name of file containing metadata
    @param train_dataset: dataset used for training/calculating clusters
    @param number_input_frames: number of video frames
    @return: Returns map of prototypes, mapping from ef-cluster-index to a list
    of prototypes (one for each video-cluster belonging to that ef-cluster)
    """
    # here prototypes correspond to cluster centers of video clusters
    prototypes = {}
    for i in range(num_ef_clusters):
        # read prototypes of ith ef-cluster: list of Video-Instances
        prototypes[i] = rh.read_video_cluster_centers(
            Path(video_clusters_directory,
                 'cluster_centers_video_' + str(i) + '.txt'))
    # get original videos of prototypes:
    prototypes = get_videos_of_prototypes(prototypes, metadata_filename,
                                          train_dataset, number_input_frames)
    return prototypes


def get_kmeans_prototype_features(num_ef_clusters, video_clusters_directory):
    """
    Calculate prototypes of video clustering created by kmeans, where a
    prototype is defined as the features of the cluster center.
    @param num_ef_clusters: number of clusters produced while clustering by EF
    @param video_clusters_directory: directory containing video cluster labels
    @return: Returns map of prototypes, mapping from ef-cluster-index to a list
    of prototypes (one for each video-cluster belonging to that ef-cluster)
    """
    # here prototypes correspond to features of cluster centers
    prototypes = {}
    for i in range(num_ef_clusters):
        print(i)
        # read prototypes of ith ef-cluster: list of Video-Instances
        prototypes[i] = rh.read_video_cluster_centers(
            Path(video_clusters_directory,
                 'cluster_centers_video_' + str(i) + '.txt'), video_known=False)
        print(len(prototypes[i]))
    return prototypes


def get_kmeans_prototype_videos(num_ef_clusters, video_features_directory,
                                video_clusters_directory, metadata_filename,
                                train_dataset, number_input_frames,
                                similarity_measure='euclidean'):
    """
    Calculate prototypes of video clustering created by kmeans, where a
    prototype is defined as the video which extracted features are most similar
    to cluster mean features.
    @param num_ef_clusters: number of clusters produced while clustering by EF
    @param video_features_directory: directory containing video feature files
    @param video_clusters_directory: directory containing video cluster labels
    @param metadata_filename: name of file containing metadata
    @param train_dataset: tf_records of train data
    @param number_input_frames: number of video frames
    @param similarity_measure: measure to be used for calculating the similarity
    between video features and cluster mean features ('euclidean', 'cosine')
    @return: Returns a map of prototypes, mapping from ef-cluster-index to a list
    of prototypes (one for each video-cluster belonging to that ef-cluster)
    """
    # iterate over all ef_clusters:
    #   iterate over all video_clusters in that ef_cluster:
    #       calculate prototype of the video_cluster
    # store prototypes in map: ef(-range) -> [prototypes]
    # (one prototype per video cluster)

    similarity_measures = ['euclidean', 'cosine']
    if similarity_measure not in similarity_measures:
        raise ValueError("ERROR: similarity measure has to be in",
                         similarity_measures)
    prototypes = {}
    for i in range(num_ef_clusters):
        prototypes[i] = []
        # read cluster centers of ith ef-cluster: list of Video-Instances
        cluster_centers = rh.read_video_cluster_centers(
            Path(video_clusters_directory,
                 'cluster_centers_video_' + str(i) + '.txt'))
        # get labels of video_clustering
        cluster_labels, actual_efs, file_names = rh.read_cluster_labels(
            Path(video_clusters_directory,
                 'cluster_labels_video_' + str(i) + '.txt'))
        cluster_features = rh.read_extracted_features(
            Path(video_features_directory,
                 'extracted_video_features_' + str(i) + '.txt'))
        num_video_clusters = max(cluster_labels) + 1
        for j in range(num_video_clusters):
            # indices of all echocardiograms contained in jth video cluster of ith ef-cluster
            videos_in_cluster = [k for k in range(len(cluster_labels))
                                 if cluster_labels[k] == j]
            # get extracted features of this subcluster
            video_cluster_features = [cluster_features[i] for i in videos_in_cluster]
            # for each (not actually existing) cluster center:
            # get closest existing instance according to similarity measure
            closest_video_index, closest_video = get_most_similar_instance(
                cluster_centers[j].features, video_cluster_features)
            prototypes[i].append(
                rh.Video(closest_video,
                         actual_efs[videos_in_cluster[closest_video_index]],
                         file_names[videos_in_cluster[closest_video_index]]))
    # get original videos of prototypes:
    prototypes = get_videos_of_prototypes(prototypes, metadata_filename,
                                          train_dataset, number_input_frames)
    return prototypes


def get_videos_of_prototypes(prototypes, metadata_filename, train_dataset,
                             number_input_frames):
    # get EF of all videos
    metadata_path = Path(metadata_filename)
    file_list_data_frame = pandas.read_csv(metadata_path)
    train_files = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['FileName', 'EF', 'ESV', 'EDV']]
    train_files = train_files.append(file_list_data_frame[file_list_data_frame.Split == 'VAL'][['FileName', 'EF', 'ESV', 'EDV']])
    train_files = train_files.reset_index()

    # get indices of prototypes in file list/tf_records
    # (i: ef cluster, j: video cluster of prototype)
    indices = {}
    # save indices (in file list) in list and sort list
    # (in order to make only one iteration over the whole train set while
    # looking for the corresponding training instances)
    all_indices = []
    # add to prototypes its complete videos (not only extracted features)
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            file_name = prototypes[i][j].file_name
            index = train_files.index[train_files.FileName == file_name].tolist()
            index = index[0]
            all_indices.append(index)
            indices[index] = []
            indices[index].append(i)
            indices[index].append(j)
    print(len(indices), "indices found")

    all_indices.sort()
    print("Indices: ", all_indices)
    i = 0  # counter for iterating over train dataset
    k = 0  # counter for iterating over all_indices
    for video, y in train_dataset:
        if k == all_indices[i]:
            prototypes[indices[k][0]][indices[k][1]].video = \
                video[:, :number_input_frames, :, :, :].numpy().flatten()
            i += 1
            if i >= len(all_indices):
                break
        k += 1
    return prototypes


def get_most_similar_instance(center_features, cluster_features):
    # get prototype: video which is the closest to cluster mean/center features
    prototype_index = 0
    prototype = cluster_features[prototype_index]
    min_dist = np.linalg.norm(np.array(center_features) - np.array(prototype))
    for i, video in enumerate(cluster_features):
        current_dist = np.linalg.norm(np.array(center_features) - np.array(video))
        if current_dist < min_dist:
            prototype = video
            prototype_index = i
            min_dist = current_dist
    return prototype_index, prototype


def calculate_mean(video_features):
    mean_features = []
    num_features = len(video_features[0].features)
    for i in range(num_features):
        mean_features.append(0)
        for video in video_features:
            mean_features[i] += video.features[i]
        mean_features[i] /= len(video_features)
    return mean_features


if __name__ == '__main__':
    main()
