import getopt
import sys
import tensorflow as tf
import numpy as np
import pathlib
import pandas as pd
import cv2
from tqdm import tqdm
import preprocessing.echos_to_tf_record as echo_base


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

    generate_tf_record(input_directory, output_directory, standardisation_sample, metadata_filename)


def generate_tf_record(input_directory, output_directory, standardisation_sample, metadata_filename):
    random_seed = 5
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    needed_frames = 50

    # load videos and file_list and put into tfrecord
    file_list_data_frame = pd.read_csv(input_directory / metadata_filename, sep=";", decimal=",")

    if output_directory is None:
        output_directory = input_directory
    p = (input_directory / 'Videos').glob('*')
    video_paths = [x.name for x in p if x.is_file()]
    file_information = pd.DataFrame([(x, x.split('_')[0], x.split('_')[1]) for x in video_paths],
                                    columns=["FileName", "id", "view"])
    file_information["id"] = file_information["id"].astype(int)
    video_metadata = file_information.merge(file_list_data_frame, on="id", how="right")
    a4c_video_metadata = video_metadata[video_metadata["view"] == "a4c"]
    a4c_video_metadata.reset_index(inplace=True, drop=True)
    train_folder = output_directory / 'tf_record' / 'train'
    test_folder = output_directory / 'tf_record' / 'test'
    validation_folder = output_directory / 'tf_record' / 'validation'

    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    train_samples, validation_samples, test_samples = np.split(a4c_video_metadata.sample(frac=1, replace=False,
                                                                                         random_state=random_seed),
                                                               [int(train_ratio * len(a4c_video_metadata)),
                                                                int((train_ratio + validation_ratio) * len(
                                                                    a4c_video_metadata))])
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
        mean, std = calculate_train_mean_and_std(input_directory, train_samples.FileName, standardisation_sample)
        echo_base.save_metadata(output_directory, needed_frames, width, height, mean, std, number_of_test_samples,
                                number_of_train_samples,
                                number_of_validation_samples)


def create_tf_record(input_directory, output_file, samples, needed_frames=50):
    number_used_videos = 0
    file_limit = 10
    chunk_size = int(len(samples) / file_limit)
    for index in tqdm(range(file_limit), file=sys.stdout):
        with tf.io.TFRecordWriter(str(output_file).format(index)) as writer:
            start = index * chunk_size
            end = start + chunk_size
            if index + 1 == file_limit:
                end = len(samples)
            for file_name, ejection_fraction, e_e_prime, quality in zip(samples.FileName[start: end],
                                                                        samples.EF[start: end],
                                                                        samples.E_E_prime_Ratio[start:end],
                                                                        samples.Quality[start:end]):
                video = echo_base.load_video(str(input_directory / 'Videos' / file_name), needed_frames)
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


def calculate_train_mean_and_std(input_directory, file_names, standardisation_sample=None):
    n = 0
    mean = 0
    M2 = 0
    if standardisation_sample is None:
        standardisation_sample = len(file_names)
    filenames_to_use = file_names[0:standardisation_sample]
    for file_name in tqdm(filenames_to_use, file=sys.stdout):
        video = cv2.VideoCapture(str(input_directory / 'Videos' / file_name))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in range(frame_count):
            ret, frame = video.read()
            if frame.shape[2] > 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape
            for i in range(height):
                for j in range(width):
                    x = frame.item(i, j)
                    n = n + 1
                    delta = x - mean
                    mean = mean + delta / n
                    M2 = M2 + delta * (x - mean)
    var = M2 / (n - 1)
    return mean, np.sqrt(var)


if __name__ == '__main__':
    main(sys.argv[1:])
