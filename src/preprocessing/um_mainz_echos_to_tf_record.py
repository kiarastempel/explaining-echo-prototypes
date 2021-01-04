import argparse
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import echos_to_tf_record as echo_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data', help="Directory with the echocardiograms.")
    parser.add_argument('-o', '--output_directory', help="Directory to save the TFRecord in")
    parser.add_argument('-s', '--standardization_size', type=int, help="Number of videos "
                        "for calculating the mean and standard deviation.")
    parser.add_argument('-m', '--metadata_filename', default='FileList.csv', help="Name of the metadata file.")
    parser.add_argument('--needed_frames', default=50, type=int, help="Number of minimum frames required for an echo "
                                                                      "to be considered.")
    args = parser.parse_args()

    if args.output_directory is None:
        args.output_directory = args.input_directory

    generate_tf_record(args.input_directory, args.output_directory, args.standardization_size,
                       args.metadata_filename, args.needed_frames)


def generate_tf_record(input_directory, output_directory, standardisation_sample, metadata_filename, needed_frames):
    random_seed = 5
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # load videos and file_list and put into tfrecord
    input_path = Path(input_directory)
    file_list_data_frame = pd.read_csv(input_path / metadata_filename, sep=";", decimal=",")

    p = (input_path / 'Videos').glob('*')
    video_paths = [x.name for x in p if x.is_file()]
    file_information = pd.DataFrame([(x, x.split('_')[0], x.split('_')[1]) for x in video_paths],
                                    columns=["FileName", "id", "view"])
    file_information["id"] = file_information["id"].astype(int)
    video_metadata = file_information.merge(file_list_data_frame, on="id", how="right")
    video_metadata.dropna(inplace=True)
    a4c_video_metadata = video_metadata[video_metadata["view"] == "a4c"]
    a2c_video_metadata = video_metadata[video_metadata["view"] == "a2c"]
    psax_video_metadata = video_metadata[video_metadata["view"] == "psax"]

    for view_metadata in (a4c_video_metadata, a2c_video_metadata, psax_video_metadata):
        view_metadata.reset_index(inplace=True, drop=True)

    train_ratio = 0.7
    validation_ratio = 0.15

    output_path = Path(output_directory)
    for view_metadata, view in zip((a4c_video_metadata, a2c_video_metadata, psax_video_metadata, video_metadata),
                                   ('a4c', 'a2c', 'psax', 'all')):
        train_folder = output_path / 'tf_records' / view / 'train'
        test_folder = output_path / 'tf_records' / view / 'test'
        validation_folder = output_path / 'tf_records' / view / 'validation'

        unique_ids = view_metadata['id'].unique()
        np.random.shuffle(unique_ids)

        train_ids, validation_ids, test_ids = np.split(unique_ids,
                                                       [int(train_ratio * len(unique_ids)),
                                                        int((train_ratio + validation_ratio) * len(
                                                            unique_ids))])
        train_samples = view_metadata[view_metadata['id'].isin(train_ids)]
        validation_samples = view_metadata[view_metadata['id'].isin(test_ids)]
        test_samples = view_metadata[view_metadata['id'].isin(validation_ids)]
        test_folder.mkdir(parents=True, exist_ok=True)
        train_folder.mkdir(exist_ok=True)
        validation_folder.mkdir(exist_ok=True)

        width, height = echo_base.extract_metadata(train_samples.FileName[1], input_path)

        print(f'Create train record for {view} echocardiograms.')
        number_of_train_samples = create_tf_record(input_path, train_folder / 'train_{}.tfrecord', train_samples,
                                                   needed_frames)

        print(f'Create test record  for {view} echocardiograms.')
        number_of_test_samples = create_tf_record(input_path, test_folder / 'test_{}.tfrecord', validation_samples,
                                                  needed_frames)

        print(f'Create validation record  for {view} echocardiograms')
        number_of_validation_samples = create_tf_record(input_path, validation_folder / 'validation_{}.tfrecord',
                                                        test_samples, needed_frames)

        metadata_file_path = output_path / 'tf_records' / view / 'metadata.json'
        if not metadata_file_path.is_file():
            print('Calculate mean and standard deviation.')
            mean, std = echo_base.calculate_train_mean_and_std(input_path, train_samples.FileName,
                                                               standardisation_sample)
            echo_base.save_metadata(metadata_file_path, needed_frames, width, height, mean, std, number_of_test_samples,
                                    number_of_train_samples,
                                    number_of_validation_samples)


def create_tf_record(input_directory, output_file, samples, needed_frames=50):
    number_used_videos = 0
    file_limit = 10
    chunk_size = int(len(samples) / file_limit)
    for index in tqdm(range(file_limit), file=sys.stdout):
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        writer = tf.io.TFRecordWriter(str(output_file).format(index), options=options)
        with writer:
            start = index * chunk_size
            end = start + chunk_size
            if index + 1 == file_limit:
                end = len(samples)
            for file_name, ejection_fraction, e_e_prime, quality in zip(samples.FileName[start: end],
                                                                        samples.EF[start: end],
                                                                        samples.E_E_prime_Ratio[start:end],
                                                                        samples.Quality[start:end]):
                video = echo_base.load_video(input_directory / 'Videos' / file_name, needed_frames)
                if video is not None:
                    number_used_videos += 1
                    writer.write(serialise_example(video, ejection_fraction, e_e_prime, quality))
    return number_used_videos


def serialise_example(video, ejection_fraction, e_e_prime, quality):
    feature = {
        'frames': echo_base.bytes_list_feature(video),
        'ejection_fraction': echo_base.float_feature(ejection_fraction),
        'e_e_prime': echo_base.float_feature(e_e_prime),
        'quality': echo_base.int64_feature(quality),
        'number_of_frames': echo_base.int64_feature(len(video))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    main()
