import getopt
import sys
import tensorflow as tf
import numpy as np
import pathlib
import pandas
from tqdm import tqdm
import echos_to_tf_record as echo_base


def main(argv):
    input_directory = ''
    output_directory = None
    standardisation_sample = None
    found_input = False
    found_metadata_filename = False
    metadata_filename = None

    try:
        opts, args = getopt.getopt(argv, "i:o:s:m:", ["input=", "output=", "standard_size=", "metadata_filename="])
    except getopt.GetoptError:
        print('test.py -i <input_directory> -o <output_directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_directory = pathlib.Path(arg)
            found_input = True
        elif opt in ("-o", "--output"):
            output_directory = pathlib.Path(arg)
        elif opt in ("-s", "--standard_size"):
            standardisation_sample = int(arg)
        elif opt in ("-m", "--metadata_filename"):
            metadata_filename = arg
            found_metadata_filename = True
    if not (found_input and found_metadata_filename):
        print('Input directory (-i) and metadata_filename (-m) are required.')
        sys.exit(2)

    generate_tf_record(input_directory, output_directory, metadata_filename, standardisation_sample)


def generate_tf_record(input_directory, output_directory, metadata_filename, standardisation_sample):
    tf.random.set_seed(5)
    np.random.seed(5)
    needed_frames = 50

    # load videos and file_list and put into tfrecord
    file_list_data_frame = pandas.read_csv(input_directory / metadata_filename)
    train_samples = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['FileName', 'EF']]
    validation_samples = file_list_data_frame[file_list_data_frame.Split == 'VAL'][['FileName', 'EF']]
    test_samples = file_list_data_frame[file_list_data_frame.Split == 'TEST'][['FileName', 'EF']]

    train_samples = train_samples.sample(frac=1)
    test_samples = test_samples.sample(frac=1)
    validation_samples = validation_samples.sample(frac=1)

    if output_directory is None:
        output_directory = input_directory

    train_folder = output_directory / 'tf_record' / 'train'
    test_folder = output_directory / 'tf_record' / 'test'
    validation_folder = output_directory / 'tf_record' / 'validation'

    test_folder.mkdir(parents=True, exist_ok=True)
    train_folder.mkdir(exist_ok=True)
    validation_folder.mkdir(exist_ok=True)

    width, height = echo_base.extract_metadata(train_samples.FileName[1], input_directory)

    print('Create train record.')
    number_of_train_samples = create_tf_record(input_directory, train_folder / 'train_{}.tfrecord', train_samples,
                                               needed_frames)

    print('Create test record.')
    number_of_test_samples = create_tf_record(input_directory, test_folder / 'test_{}.tfrecord', validation_samples,
                                              needed_frames)

    print('Create validation record.')
    number_of_validation_samples = create_tf_record(input_directory, validation_folder / 'validation_{}.tfrecord',
                                                    test_samples, needed_frames)

    if not (output_directory / 'tf_record' / 'metadata.json').is_file():
        print('Calculate mean and standard deviation.')
        mean, std = echo_base.calculate_train_mean_and_std(input_directory, train_samples.FileName,
                                                           standardisation_sample)
        echo_base.save_metadata(output_directory, needed_frames, width, height, mean, std, number_of_test_samples,
                                number_of_train_samples,
                                number_of_validation_samples)


def create_tf_record(input_directory, output_file, samples, needed_frames):
    number_used_videos = 0
    file_limit = 10
    chunk_size = int(len(samples) / file_limit)
    for index in tqdm(range(file_limit), file=sys.stdout):
        with tf.io.TFRecordWriter(str(output_file).format(index)) as writer:
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
    main(sys.argv[1:])
