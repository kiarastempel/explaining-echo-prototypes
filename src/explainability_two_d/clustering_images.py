import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
import read_helpers as rh
import clustering as cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with still images.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the image cluster labels in")
    parser.add_argument('-if', '--image_features_directory',
                        default='../../data/image_features',
                        help='Directory with image features')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-cl', '--volume_clusters_file',
                        default='../../data/clustering_volume/cluster_labels_esv.txt',
                        help='Path to file containing volume cluster labels')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    parser.add_argument('-n', '--max_n_clusters', default=100, type=int,
                        help="Maximum number of clusters to be evaluated.")
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'image_clusters')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    # get data of volume clustering
    volume_cluster_labels, actual_volumes, file_names = rh.read_cluster_labels(args.volume_clusters_file)
    print('Data loaded')

    cluster_by_latent_features(volume_cluster_labels,
                               actual_volumes,
                               file_names,
                               args.image_features_directory,
                               output_directory,
                               n=args.max_n_clusters,
                               standardize=True,
                               normalize=False)


def cluster_by_latent_features(volume_cluster_labels, actual_volumes, file_names,
                               image_features_directory, output_directory,
                               n=100,
                               standardize=True, normalize=False,
                               pca=False, pca_components=3):
    # for standardization/normalization: num_instances * num_features
    all_extracted_features = []
    # for clustering each volume cluster: num_clusters * cluster_size * num_features
    extracted_features_per_cluster = []
    num_volume_clusters = max(volume_cluster_labels) + 1

    # get extracted features for all volume clusters
    for i in range(num_volume_clusters):
        file_path = Path(image_features_directory,
                         'extracted_image_features_' + str(i) + '.txt')
        extracted_features = rh.read_extracted_features(file_path)
        all_extracted_features.extend(extracted_features)
        extracted_features_per_cluster.append(extracted_features)
        print("Image features of volume cluster", i, "read")
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
        for i in range(num_volume_clusters):
            file_path = Path(image_features_directory,
                             'extracted_image_features_' + str(i) + '.txt')
            extracted_features = rh.read_extracted_features(file_path)
            all_extracted_features.extend(extracted_features)
            extracted_features_per_cluster.append(extracted_features)

        pca = PCA(n_components=pca_components)
        extracted_features_per_cluster = transform(
            pca, all_extracted_features, extracted_features_per_cluster)
        explained_variance = pca.explained_variance_ratio_
        print("Explained variance:", explained_variance)

    # cluster images in each volume cluster using extracted features
    for i in range(num_volume_clusters):
        # indices of all still images contained in ith volume cluster
        images_in_cluster = [j for j in range(len(volume_cluster_labels))
                             if volume_cluster_labels[j] == i]
        # K-Medoids
        max_n_clusters = min(n, len(images_in_cluster) - 2)
        out_file_path = Path(output_directory, 'cluster_labels_image_' + str(i) + '.txt')
        out_file_path_centers = Path(output_directory, 'cluster_centers_image_' + str(i) + '.txt')
        # only one image in volume cluster -> put it in the only image cluster
        if max_n_clusters < 2:
            cluster_labels = [0]
            cluster_centers = [extracted_features_per_cluster[i][0]]
        else:
            cluster_labels, cluster_centers = cl.compare_n_clusters_k_medoids(
                extracted_features_per_cluster[i], max_n_clusters, plot=True,
                plot_name="volume_cluster_" + str(i))[0:2]

        # get indices of corresponding echocardiograms
        cluster_centers_indices = get_image_cluster_centers_indices(
            cluster_centers, extracted_features_per_cluster[i])
        print("kmedoids", cluster_labels)
        print(images_in_cluster)

        # save labels and centers
        with open(out_file_path, "w") as txt_file:
            for j in range(len(cluster_labels)):
                txt_file.write(str(cluster_labels[j]) + " "
                               + str(actual_volumes[images_in_cluster[j]]) + " "
                               + str(file_names[images_in_cluster[j]]) + "\n")
        with open(out_file_path_centers, "w") as txt_file:
            for j in range(len(cluster_centers)):
                txt_file.write(str(j) + " "
                               + str(actual_volumes[images_in_cluster[cluster_centers_indices[j]]]) + " "
                               + str(file_names[images_in_cluster[cluster_centers_indices[j]]]) + " "
                               + str(np.array(original_extracted_features_per_cluster[i][cluster_centers_indices[j]])) + "\n")
        print("Clustering of volume cluster", i, "finished", "\n")


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
    return extracted_features_per_cluster_ft


def get_image_cluster_centers_indices(cluster_centers, extracted_features):
    indices = []
    for c in cluster_centers:
        # find echocardiographic still image that corresponds to features of c
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
