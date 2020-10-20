import getopt
import sys
import tensorflow as tf
import numpy as np
import pathlib
import pandas
import cv2
import json
from tqdm import tqdm


def main(argv):
    input_directory = ''
    output_directory = ''
    standardisation_sample = None
    found_input = False
    found_output = False
    try:
        opts, args = getopt.getopt(argv, "i:o:s:", ["input=", "output=", "standard_size="])
    except getopt.GetoptError:
        print('test.py -i <input_directory> -o <output_directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_directory = pathlib.Path(arg)
            found_input = True
        elif opt in ("-o", "--output"):
            output_directory = pathlib.Path(arg)
            found_output = True
        elif opt in ("-s", "--standard_size"):
            standardisation_sample = int(arg)
    if not (found_input and found_output):
        print('Input directory (-i) and output directory are (-o) are required')
        sys.exit(2)

    generate_tf_record(input_directory, output_directory, standardisation_sample)


def generate_tf_record(input_directory, output_directory, standardisation_sample):
    tf.random.set_seed(5)
    np.random.seed(5)

    # load videos and file_list and put into tfrecord
    file_list_data_frame = pandas.read_csv(input_directory / 'FileList.csv')
    train_samples = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['FileName', 'EF']]
    validation_samples = file_list_data_frame[file_list_data_frame.Split == 'VAL'][['FileName', 'EF']]
    test_samples = file_list_data_frame[file_list_data_frame.Split == 'TEST'][['FileName', 'EF']]

    train_samples = train_samples.sample(frac=1)
    test_samples = test_samples.sample(frac=1)
    validation_samples = validation_samples.sample(frac=1)

    train_folder = output_directory / 'train'
    test_folder = output_directory / 'test'
    validation_folder = output_directory / 'validation'
    test_folder.mkdir(parents=True, exist_ok=True)
    train_folder.mkdir(exist_ok=True)
    validation_folder.mkdir(exist_ok=True)

    fps, width, height = extract_metadata(train_samples.FileName[1], input_directory)

    print('Calculate mean and standard deviation.')
    mean, std = calculate_train_mean_and_std(input_directory, train_samples.FileName, standardisation_sample)
    print('Create train record.')
    number_of_train_samples = create_tf_record(input_directory, train_folder / 'train_{}.tfrecord', train_samples, fps,
                                               width, height,)

    print('Create test record.')
    number_of_test_samples = create_tf_record(input_directory, test_folder / 'test_{}.tfrecord', validation_samples,
                                              fps, width, height,)

    print('Create validation record.')
    number_of_validation_samples = create_tf_record(input_directory, validation_folder / 'validation_{}.tfrecord',
                                                    test_samples, fps, width, height,)

    save_metadata(output_directory, fps, width, height, mean, std, number_of_test_samples, number_of_train_samples,
                  number_of_validation_samples)


def extract_metadata(file_name, input_directory):
    video = cv2.VideoCapture(str(input_directory / 'Videos' / file_name))
    frames_per_second = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frames_per_second, frame_width, frame_height


def save_metadata(output_directory, frames_per_second, frame_width, frame_height, mean, std, number_of_test_samples,
                  number_of_train_samples, number_of_validation_samples):
    metadata = {'metadata': {
        'frames_per_second': frames_per_second,
        'frame_height': frame_height,
        'frame_width': frame_width,
        'number_of_test_samples': number_of_test_samples,
        'number_of_train_samples': number_of_train_samples,
        'number_of_validation_samples': number_of_validation_samples,
        'mean': mean,
        'std': std,
        'channels': 1
    }
    }

    with open(output_directory / 'metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)


def create_tf_record(input_directory, output_file, samples, width, height, needed_frames=50):
    number_used_videos = 0
    file_limit = 10
    chunk_size = int(len(samples) / file_limit)
    for index in tqdm(range(file_limit), file=sys.stdout):
        with tf.io.TFRecordWriter(str(output_file).format(index)) as writer:
            start = index * chunk_size
            for file_name, ejection_fraction in zip(samples.FileName[start: start + chunk_size],
                                                    samples.EF[start: start + chunk_size]):
                video = load_video(str(input_directory / 'Videos' / file_name), needed_frames)
                if video is not None:
                    number_used_videos += 1
                    writer.write(serialise_example(video, ejection_fraction, width, height))
    return number_used_videos


def load_video(file_name, needed_frames):
    video = cv2.VideoCapture(file_name)
    frame_list = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if needed_frames > frame_count:
        return None
    for _ in range(frame_count):
        ret, frame = video.read()
        if frame.shape[2] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_list.append(frame.tobytes())
    video.release()
    return frame_list


def serialise_example(video, ejection_fraction, width, height):
    feature = {
        'frames': _bytes_list_feature(video),
        'ejection_fraction': _float_feature(ejection_fraction),
        'number_of_frames': _int64_feature(len(video)),
        'width': _int64_feature(width),
        'height': _int64_feature(height)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def calculate_train_mean_and_std(input_directory, file_names, standardisation_sample):
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


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


if __name__ == '__main__':
    main(sys.argv[1:])
