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
    found_input = False
    found_output = False
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["input=", "output="])
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
    if not (found_input and found_output):
        print('Input directory (-i) and output directory are (-o) are required')
        sys.exit(2)

    generate_tf_record(input_directory, output_directory)


def generate_tf_record(input_directory, output_directory):
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

    number_of_train_samples = len(train_samples)
    number_of_validation_samples = len(validation_samples)
    number_of_test_samples = len(test_samples)
    train_folder = output_directory / 'train'
    test_folder = output_directory / 'test'
    validation_folder = output_directory / 'validation'
    test_folder.mkdir(parents=True, exist_ok=True)
    train_folder.mkdir(exist_ok=True)
    validation_folder.mkdir(exist_ok=True)

    fps, width, height = extract_metadata(train_samples.FileName[1], input_directory, output_directory)
    save_metadata(output_directory, fps, width, height, number_of_test_samples, number_of_train_samples,
                  number_of_validation_samples)
    print('Create train record.')
    create_tf_record(input_directory, train_folder / 'train_{}.tfrecord', train_samples, fps, width, height)

    print('Create test record.')
    create_tf_record(input_directory, test_folder / 'test_{}.tfrecord', validation_samples, fps, width, height)

    print('Create validation record.')
    create_tf_record(input_directory, validation_folder / 'validation_{}.tfrecord', test_samples, fps, width, height)


def extract_metadata(file_name, input_directory, output_directory):
    video = cv2.VideoCapture(str(input_directory / 'Videos' / file_name))
    frames_per_second = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frames_per_second, frame_width, frame_height


def save_metadata(output_directory, frames_per_second, frame_width, frame_height, number_of_test_samples,
                  number_of_train_samples, number_of_validation_samples):
    metadata = {'metadata': {
        'frame_count': frames_per_second,
        'frame_height': frame_height,
        'frame_width': frame_width,
        'number_of_test_samples': number_of_test_samples,
        'number_of_train_samples': number_of_train_samples,
        'number_of_validation_samples': number_of_validation_samples
        }
    }

    with open(output_directory / 'metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)


def create_tf_record(input_directory, output_file, train_samples, fps, width, height ):
    file_limit = 10
    chunk_size = int(len(train_samples) / file_limit)
    for index in tqdm(range(file_limit)):
        with tf.io.TFRecordWriter(str(output_file).format(index)) as writer:
            start = index * chunk_size
            for file_name, ejection_fraction in zip(train_samples.FileName[start: start + chunk_size],
                                                    train_samples.EF[start: start + chunk_size]):
                video = load_video(str(input_directory / 'Videos' / file_name))
                if video is not None:
                    writer.write(serialise_example(video, ejection_fraction, fps, width, height ))


def load_video(file_name):
    video = cv2.VideoCapture(file_name)
    frame_list = []
    needed_frames = 50
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < needed_frames:
        return None

    for i in range(needed_frames):
        ret, frame = video.read()
        frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).tobytes())
    video.release()
    return frame_list


def serialise_example(video, ejection_fraction, fps, width, height):
    feature = {
        'frames': _bytes_list_feature(video),
        'ejection_fraction': _float_feature(ejection_fraction),
        'fps': _int64_feature(fps),
        'width': _int64_feature(width),
        'height': _int64_feature(height)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


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
