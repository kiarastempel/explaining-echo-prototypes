import getopt
import sys
import tensorflow as tf
import numpy as np
import pathlib
import pandas
import cv2


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

    create_tf_record(input_directory, output_directory / 'train.tfrecord', train_samples)
    create_tf_record(input_directory, output_directory / 'validation.tfrecord', validation_samples)
    create_tf_record(input_directory, output_directory / 'test.tfrecord', test_samples)


def create_tf_record(input_directory, output_file, train_samples):
    with tf.io.TFRecordWriter(str(output_file)) as writer:
        for file_name, ejection_fraction in zip(train_samples.FileName, train_samples.EF):
            video = load_video(str(input_directory / 'Videos' / file_name))
            writer.write(serialise_example(video, ejection_fraction))


def load_video(file_name):
    video = cv2.VideoCapture(file_name)
    frame_list = []
    needed_frames = 50
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count < 50:
        needed_frames = frame_count

    for i in range(needed_frames):
        ret, frame = video.read()
        frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    video.release()
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                      for frame in frame_list]
    return encoded_frames


def serialise_example(video, ejection_fraction):
    feature = {
        'video': _bytes_list_feature(video),
        'ejection_fraction': _float_feature(ejection_fraction)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


if __name__ == '__main__':
    main(sys.argv[1:])
