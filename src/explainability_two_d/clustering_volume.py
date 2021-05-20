import argparse
from pathlib import Path
import pandas
from explainability import clustering_ef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with still images.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the cluster labels in")
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-n', '--max_n_clusters', default=100, type=int,
                        help="Maximum number of clusters to be evaluated.")
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'clustering_volume')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # get EF of all videos
    metadata_path = Path(args.input_directory, args.frame_volumes_filename)
    file_list_data_frame = pandas.read_csv(metadata_path)
    volumes = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['Image_FileName', 'Volume']]
    volumes = volumes.append(file_list_data_frame[file_list_data_frame.Split == 'VAL'][['Image_FileName', 'Volume']])
    volumes = volumes.reset_index()
    volumes['FileName'] = volumes['Image_FileName']
    volumes['EF'] = volumes['Volume']  # just to allow reuse of cluster function

    clustering_ef.cluster_by_ef(volumes, args.max_n_clusters, output_directory)


if __name__ == '__main__':
    main()
