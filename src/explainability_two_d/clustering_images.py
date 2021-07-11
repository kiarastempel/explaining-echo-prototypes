import argparse
from pathlib import Path
from explainability import clustering_videos
from explainability import read_helpers as rh


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
                        default='../../data/clustering_volume/cluster_labels_ef.txt',
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

    clustering_videos.cluster_by_videos(volume_cluster_labels,
                                        actual_volumes,
                                        file_names,
                                        args.image_features_directory,
                                        output_directory,
                                        n=args.max_n_clusters,
                                        standardize=True,
                                        normalize=False)


if __name__ == '__main__':
    main()
