import argparse
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas
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

    generate_tf_record(args.input_directory, args.output_directory, args.metadata_filename,
                       args.standardization_size, args.needed_frames)


def generate_tf_record(input_directory, output_directory, metadata_filename, standardisation_sample, needed_frames):
    tf.random.set_seed(5)
    np.random.seed(5)

    # load videos and file_list and put into tfrecord
    metadata_path = Path(input_directory, metadata_filename)
    file_list_data_frame = pandas.read_csv(metadata_path)
    train_samples = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['FileName', 'EF']]
    validation_samples = file_list_data_frame[file_list_data_frame.Split == 'VAL'][['FileName', 'EF']]
    test_samples = file_list_data_frame[file_list_data_frame.Split == 'TEST'][['FileName', 'EF']]

    train_samples = train_samples.sample(frac=1)
    test_samples = test_samples.sample(frac=1)
    validation_samples = validation_samples.sample(frac=1)

    if output_directory is None:
        output_directory = input_directory

    output_path = Path(output_directory)
    input_path = Path(input_directory)

    train_folder = output_path / 'tf_record' / 'train'
    test_folder = output_path / 'tf_record' / 'test'
    validation_folder = output_path / 'tf_record' / 'validation'

    test_folder.mkdir(parents=True, exist_ok=True)
    train_folder.mkdir(exist_ok=True)
    validation_folder.mkdir(exist_ok=True)

    width, height = echo_base.extract_metadata(train_samples.FileName[1], input_path)

    print('Create train record.')
    number_of_train_samples = create_tf_record(input_path, train_folder / 'train_{}.tfrecord.gzip', train_samples,
                                               needed_frames)

    print('Create test record.')
    number_of_test_samples = create_tf_record(input_path, test_folder / 'test_{}.tfrecord.gzip', validation_samples,
                                              needed_frames)

    print('Create validation record.')
    number_of_validation_samples = create_tf_record(input_path, validation_folder / 'validation_{}.tfrecord.gzip',
                                                    test_samples, needed_frames)

    metadata_file_path = output_directory / 'tf_record' / 'metadata.json'
    if not metadata_file_path.is_file():
        print('Calculate mean and standard deviation.')
        mean, std = echo_base.calculate_train_mean_and_std(input_path, train_samples.FileName,
                                                           standardisation_sample)

        echo_base.save_metadata(metadata_file_path, needed_frames, width, height, mean, std, number_of_test_samples,
                                number_of_train_samples,
                                number_of_validation_samples)


def create_tf_record(input_directory, output_file, samples, needed_frames):
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
            for file_name, ejection_fraction in zip(samples.FileName[start: end],
                                                    samples.EF[start: end]):
                video = echo_base.load_video(str(input_directory / 'Videos' / file_name), needed_frames)
                if video is not None:
                    number_used_videos += 1
                    writer.write(serialise_example(video, ejection_fraction))
    return number_used_videos


def serialise_example(video, ejection_fraction):
    feature = {
        'frames': echo_base.bytes_list_feature(video),
        'ejection_fraction': echo_base.float_feature(ejection_fraction),
        'number_of_frames': echo_base.int64_feature(len(video))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    main()
