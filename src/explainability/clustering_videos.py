import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
import numpy as np
import explainability.read_helpers as rh
import explainability.clustering as cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the TFRecord files.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the video cluster labels in")
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('-cl', '--ef_clusters_file',
                        default='../../data/clustering_ef/cluster_labels_ef.txt',
                        help='Path to file containing ef cluster labels')
    parser.add_argument('-vf', '--video_features_directory',
                        default='../../data/video_features',
                        help='Directory with video features')
    parser.add_argument('-model_path', required=True)
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'video_clusters')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    # get data of ef clustering
    ef_cluster_labels, actual_efs, file_names = rh.read_cluster_labels(
        args.ef_clusters_file)

    cluster_by_videos(ef_cluster_labels, actual_efs, file_names,
                      args.video_features_directory,
                      output_directory, standardize=True, normalize=False)


def cluster_by_videos(ef_cluster_labels, actual_efs, file_names,
                      video_features_directory,
                      output_directory, standardize=True, normalize=False,
                      pca=False):

    # for standardization/normalization: num_instances * num_features
    all_extracted_features = []
    # for clustering each ef-cluster: num_clusters * cluster_size * num_features
    extracted_features_per_cluster = []
    num_ef_clusters = max(ef_cluster_labels) + 1

    # get extracted features for all ef-clusters
    for i in range(num_ef_clusters):
        file_path = Path(video_features_directory,
                         'extracted_video_features_' + str(i) + '.txt')
        extracted_features = rh.read_extracted_features(file_path)
        all_extracted_features.extend(extracted_features)
        extracted_features_per_cluster.append(extracted_features)
        print("Videos features of ef-cluster", i, "read")
    original_extracted_features_per_cluster = extracted_features_per_cluster

    # normalize/standardize features
    if normalize:
        print('normalizing')
        extracted_features_per_cluster = transform(
            Normalizer(), all_extracted_features,
            extracted_features_per_cluster)
    if standardize:
        print('standardizing')
        extracted_features_per_cluster = transform(
            StandardScaler(), all_extracted_features,
            extracted_features_per_cluster)
    # dimensionality reduction
    if pca:
        for i in range(num_ef_clusters):
            file_path = Path(video_features_directory,
                             'extracted_video_features_' + str(i) + '.txt')
            extracted_features = rh.read_extracted_features(file_path)
            all_extracted_features.extend(extracted_features)
            extracted_features_per_cluster.append(extracted_features)

        pca = PCA(n_components=3)
        extracted_features_per_cluster = transform(
            pca, all_extracted_features, extracted_features_per_cluster)
        explained_variance = pca.explained_variance_ratio_
        print("##", explained_variance)

    # cluster videos in each ef-cluster using extracted features
    for i in range(num_ef_clusters):
        # indices of all echocardiograms contained in ith ef-cluster
        videos_in_cluster = [j for j in range(len(ef_cluster_labels))
                             if ef_cluster_labels[j] == i]
        # K-Medoids
        max_n_clusters = min(100, len(videos_in_cluster))
        out_file_path = Path(output_directory, 'cluster_labels_video_' + str(i) + '.txt')
        out_file_path_centers = Path(output_directory, 'cluster_centers_video_' + str(i) + '.txt')
        # only one video in ef-cluster -> put it in the only video cluster
        if max_n_clusters < 2:
            cluster_labels = [0]
            cluster_centers = [extracted_features_per_cluster[i][0]]
        else:
            cluster_labels, cluster_centers = cl.compare_n_clusters_k_medoids(
                extracted_features_per_cluster[i], max_n_clusters, plot=True,
                plot_name="cluster_labels_video_" + str(i) + ".png")[0:2]

        # get indices of corresponding echocardiograms
        cluster_centers_indices = get_video_cluster_centers_indices(
            cluster_centers, extracted_features_per_cluster[i])
        print("kmedoids", cluster_labels)
        print(videos_in_cluster)

        # save labels and centers
        with open(out_file_path, "w") as txt_file:
            for j in range(len(cluster_labels)):
                txt_file.write(str(cluster_labels[j]) + " "
                               + str(actual_efs[videos_in_cluster[j]]) + " "
                               + str(file_names[videos_in_cluster[j]]) + "\n")
        with open(out_file_path_centers, "w") as txt_file:
            for j in range(len(cluster_centers)):
                txt_file.write(str(j) + " "
                               + str(actual_efs[videos_in_cluster[cluster_centers_indices[j]]]) + " "
                               + str(file_names[videos_in_cluster[cluster_centers_indices[j]]]) + " "
                               + str(np.array(original_extracted_features_per_cluster[i][cluster_centers_indices[j]])) + "\n")
        print("Clustering of ef-cluster", i, "finished", "\n")


def transform(transformer, all_extracted_features, extracted_features_per_cluster):
    # transform all features
    all_extracted_features_ft = transformer.fit_transform(all_extracted_features)
    # copy transformed features to corresponding indices of features per cluster
    extracted_features_per_cluster_ft = []
    offset = 0
    for cf in extracted_features_per_cluster:
        extracted_features_per_cluster_ft.append(
            all_extracted_features_ft[offset:offset + len(cf)])
        offset += len(cf)
    # print(([i[0] for i in extracted_features_per_cluster[0][:10]]), "\n",
    #       ([i[0] for i in extracted_features_per_cluster_ft[0][:10]]))
    return extracted_features_per_cluster_ft


def get_video_cluster_centers_indices(cluster_centers, extracted_features):
    indices = []
    for c in cluster_centers:
        # find echocardiogram that corresponds to features of c
        for i, features in enumerate(extracted_features):
            if (features == c).all():
                indices.append(i)
    return indices


def visualize_feature_distribution(all_extracted_features):
    x = []
    for f in all_extracted_features:
        x.extend(f)
    plt.hist(x)
    plt.show()


if __name__ == '__main__':
    main()

