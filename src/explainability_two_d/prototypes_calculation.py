import argparse
import numpy as np
import os
from pathlib import Path
from explainability import prototypes_calculation_videos
from two_D_resnet import get_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with still images.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save prototypes and evaluations in")
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
        output_directory = Path(args.input_directory, 'image_clusters')
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
    num_volume_clusters = int(len(os.listdir(image_clusters_directory)) / 2)
    prototypes, ef_cluster_sizes, ef_cluster_means, ef_cluster_stds, video_cluster_sizes, video_cluster_means, video_cluster_stds = \
        prototypes_calculation_videos.get_kmedoids_prototype_videos(
            num_volume_clusters,
            image_clusters_directory,
            None,
            None,
            None,
            get_videos=False
        )

    print("--------------")
    for i in range(len(ef_cluster_sizes)):
        print("EF Cluster", i, "size", ef_cluster_sizes[i], "mean",
              ef_cluster_means[i], "stds", ef_cluster_stds[i])
        for j in range(len(video_cluster_sizes[i])):
            print("Video Cluster", j, "size", video_cluster_sizes[i][j], "mean", video_cluster_means[i][j], "stds", video_cluster_stds[i][j])
        print("--------------")

    if get_images:
        prototypes = get_images_of_prototypes(prototypes, still_images, file_names)

    print("Prototypes calculated")
    with open(output_file, "w") as txt_file:
        for cp in prototypes:
            for i, p in enumerate(prototypes[cp]):
                line = str(cp) + ' ' \
                       + str(i) + ' ' \
                       + str(p.ef) + ' ' \
                       + str(p.file_name) + ' ' \
                       + str(np.array(p.features))
                if get_images:
                    line += ' ' + str(p.video)
                line += '\n'
                txt_file.write(line)


def get_images_of_prototypes(prototypes, still_images, file_names):
    file_names = list(file_names)
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            file_name = prototypes[i][j].file_name
            index = list(file_names).index(file_name)
            prototypes[i][j].video = still_images[index]
    return prototypes


if __name__ == '__main__':
    main()
