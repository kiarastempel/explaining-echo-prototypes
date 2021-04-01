import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from clustering_videos import transform
from prototypes_calculation import calculate_mean
import read_helpers as rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the TFRecord files.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the cipa video features and cluster labels in")
    parser.add_argument('-cl', '--ef_clusters_file',
                        default='../../data/clustering_ef/cluster_labels_ef.txt',
                        help='Path to file containing ef cluster labels')
    parser.add_argument('-vf', '--video_features_directory',
                        default='../../data/video_features',
                        help='Directory with video features')
    parser.add_argument('-cvc', '--cipa_video_clusters_directory',
                        default='../../data/video_clusters_cipa_original',
                        help='Directory with cipa video cluster labels')
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    if output_directory is None:
        output_directory = Path(args.input_directory)

    output_directory_cipa_features = Path(output_directory, 'video_features_cipa')
    output_directory_cipa_clusters = Path(output_directory, 'video_clusters_cipa')
    output_directory_cipa_features.mkdir(parents=True, exist_ok=True)
    output_directory_cipa_clusters.mkdir(parents=True, exist_ok=True)

    # generate_cipa_inputs(args.ef_clusters_file, args.video_features_directory, output_directory_cipa_features)
    # ... cluster each ef-cluster using cipa's java implementation ...
    save_cipa_output(args.ef_clusters_file, args.video_features_directory, args.cipa_video_clusters_directory, output_directory_cipa_clusters)


def generate_cipa_inputs(ef_clusters_file, video_features_directory, output_directory, pca=True):
    ef_cluster_labels, actual_efs, file_names = rh.read_cluster_labels(ef_clusters_file)
    num_ef_clusters = max(ef_cluster_labels) + 1
    # for dimensionality reduction: num_instances * num_features
    all_extracted_features = []
    # for clustering each ef-cluster: num_clusters * cluster_size * num_features
    extracted_features_per_cluster = []
    num_ef_clusters = max(ef_cluster_labels) + 1

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

    # generate for each ef-cluster the cipa input file (containing the features)
    for i in range(num_ef_clusters):
        save_path = Path(output_directory, 'cipa_extracted_video_features_' + str(i) + '.txt')
        generate_cipa_input(extracted_features_per_cluster[i], save_path)


def generate_cipa_input(extracted_features, file_path, pca=True):
    with open(file_path, "w") as txt_file:
        for features in extracted_features:
            for f in features:  # features[0:20]:
                txt_file.write(str(f) + " ")
            txt_file.write("\n")


def save_cipa_output(ef_clusters_file, video_features_directory, cipa_video_clusters_directory, output_directory_cipa_clusters):
    ef_cluster_labels, actual_efs, file_names = rh.read_cluster_labels(
        ef_clusters_file)
    num_ef_clusters = max(ef_cluster_labels) + 1

    for i in range(num_ef_clusters):
        file_path = Path(cipa_video_clusters_directory,
                         'cipa_extracted_video_features_' + str(i) + '.txt-Final-SyncS-ID.txt')
        cluster_labels = []
        with open(file_path, "r") as txt_file:
            for line in txt_file:
                cluster_labels.append(int(line))
        videos_in_ef_cluster = [j for j in range(len(ef_cluster_labels))
                             if ef_cluster_labels[j] == i]
        file_path = Path(video_features_directory,
                         'extracted_video_features_' + str(i) + '.txt')
        extracted_features = rh.read_extracted_features(file_path)

        out_file_path = Path(output_directory_cipa_clusters, 'cluster_labels_video_' + str(i) + '.txt')
        out_file_path_centers = Path(output_directory_cipa_clusters, 'cluster_centers_video_' + str(i) + '.txt')
        num_video_clusters = max(cluster_labels) + 1
        only_outliers = True
        with open(out_file_path, 'w') as txt_file:
            for j in range(len(cluster_labels)):
                if cluster_labels[j] != -1:
                    only_outliers = False
                    txt_file.write(str(cluster_labels[j]) + " "
                                   + str(actual_efs[videos_in_ef_cluster[j]]) + " "
                                   + str(file_names[videos_in_ef_cluster[j]]) + "\n")
        with open(out_file_path_centers, 'w') as txt_file:
            for j in range(num_video_clusters):
                videos_in_video_cluster = [k for k in range(len(cluster_labels))
                                           if cluster_labels[k] == j]
                video_features = [rh.Video(extracted_features[k], None, None) for k in videos_in_video_cluster]
                center = calculate_mean(video_features)
                txt_file.write(str(j) + " " + str(np.array(center)) + "\n")
            if only_outliers:
                for j in range(len(extracted_features)):
                    txt_file.write(str(j) + " " + str(np.array(center)) + "\n")


if __name__ == '__main__':
    main()
