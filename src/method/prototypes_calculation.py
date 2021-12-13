import argparse
import numpy as np
import os
from pathlib import Path
from model.two_d_resnet import get_data
from utils import read_helpers as rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with still images.')
    parser.add_argument('-o', '--output_directory',
                        help='Directory to save prototypes and evaluations in.')
    parser.add_argument('-p', '--prototypes_filename', default='prototypes.txt',
                        help='Name of file to save prototypes in')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-if', '--image_features_directory',
                        default='../../data/image_features',
                        help='Directory with image features')
    parser.add_argument('-ic', '--image_clusters_directory',
                        default='../../data/image_clusters',
                        help='Directory with image cluster labels')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'results')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_directory, args.prototypes_filename)
    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)

    # get train/validation/test data
    train_still_images, _, train_filenames, _, _, _, val_still_images, _, val_filenames = get_data(
        args.input_directory, frame_volumes_path, volume_type=args.volume_type)
    still_images = train_still_images.extend(val_still_images)
    file_names = train_filenames.extend(val_filenames)
    print('Data loaded')

    calculate_prototypes(still_images, file_names,
                         args.image_clusters_directory, output_file,
                         get_images=False)


def calculate_prototypes(still_images, file_names, image_clusters_directory,
                         output_file, get_images=True):
    """
    Get the still image instances that correspond to the cluster centers
    calculated using kmedoids. The prototypes are saved to files.
    @param still_images: list containing the still image frames
    @param file_names: file names of still images
    @param image_clusters_directory: directory containing the clustering
    results of kmedoids
    @param output_file: file to save the prototypes in
    @param get_images: true if the frames (consisting of pixels) of the
    prototypes should be saved in addition to the latent features
    """
    num_volume_clusters = int(len(os.listdir(image_clusters_directory)) / 2)
    prototypes, volume_cluster_sizes, volume_cluster_means, volume_cluster_stds, image_cluster_sizes, image_cluster_means, image_cluster_stds \
        = get_kmedoids_prototypes_data(
            num_volume_clusters,
            image_clusters_directory
    )

    print('--------------')
    for i in range(len(volume_cluster_sizes)):
        print('Volume cluster', i,
              ': size', volume_cluster_sizes[i],
              ', mean', volume_cluster_means[i],
              ', stds', volume_cluster_stds[i])
        for j in range(len(image_cluster_sizes[i])):
            print('Image cluster', j,
                  ': size', image_cluster_sizes[i][j],
                  ', mean', image_cluster_means[i][j],
                  ', stds', image_cluster_stds[i][j])
        print('--------------')

    if get_images:
        prototypes = get_images_of_prototypes(prototypes, still_images, file_names)

    print('Prototypes calculated')
    with open(output_file, 'w') as txt_file:
        for cp in prototypes:
            for i, p in enumerate(prototypes[cp]):
                line = str(cp) + ' ' \
                       + str(i) + ' ' \
                       + str(p.volume) + ' ' \
                       + str(p.file_name) + ' ' \
                       + str(np.array(p.features))
                if get_images:
                    line += ' ' + str(p.image)
                line += '\n'
                txt_file.write(line)


def get_images_of_prototypes(prototypes, still_images, file_names):
    """Get the real images that correspond to the file names of the
    prototypes."""
    file_names = list(file_names)
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            file_name = prototypes[i][j].file_name
            index = list(file_names).index(file_name)
            prototypes[i][j].image = still_images[index]
    return prototypes


def get_kmedoids_prototypes_data(num_volume_clusters, image_clusters_directory):
    """
    Calculate prototypes of image clustering created by kmedoids, where a
    prototype is defined as the medoid of an image cluster (i.e. it is an
    existing image).
    @param num_volume_clusters: number of clusters produced while clustering by
    volume
    @param image_clusters_directory: directory containing image cluster labels
    @return: Returns map of prototypes, mapping from volume-cluster-index to a
    list
    of prototypes (one for each image cluster belonging to that volume cluster)
    """
    # here prototypes correspond to cluster centers of image clusters
    prototypes = {}
    # means and std regarding volume
    volume_cluster_sizes = []
    volume_cluster_means = []
    volume_cluster_stds = []
    image_cluster_sizes = []
    image_cluster_means = []
    image_cluster_stds = []
    for i in range(num_volume_clusters):
        # get labels of image clustering
        cluster_labels, actual_volumes, file_names = rh.read_cluster_labels(
            Path(image_clusters_directory,
                 'cluster_labels_image_' + str(i) + '.txt'))
        num_image_clusters = max(cluster_labels) + 1
        volume_cluster_sizes.append(len(cluster_labels))
        volume_cluster_means.append(np.mean(actual_volumes))
        volume_cluster_stds.append(np.std(actual_volumes))
        image_cluster_sizes.append([])
        image_cluster_means.append([])
        image_cluster_stds.append([])
        for j in range(num_image_clusters):
            images_in_cluster = [k for k in range(len(cluster_labels))
                                 if cluster_labels[k] == j]
            volumes_of_cluster = []
            for v in images_in_cluster:
                volumes_of_cluster.append(actual_volumes[v])
            image_cluster_sizes[i].append(len(images_in_cluster))
            image_cluster_means[i].append(np.mean(volumes_of_cluster))
            image_cluster_stds[i].append(np.std(volumes_of_cluster))
        # read prototypes of ith volume cluster: list of Image-Instances
        prototypes[i] = rh.read_image_cluster_centers(
            Path(image_clusters_directory,
                 'cluster_centers_image_' + str(i) + '.txt'))

    return prototypes, volume_cluster_sizes, volume_cluster_means, volume_cluster_stds, image_cluster_sizes, image_cluster_means, image_cluster_stds


if __name__ == '__main__':
    main()
