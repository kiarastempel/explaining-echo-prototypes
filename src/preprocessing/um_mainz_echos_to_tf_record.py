import getopt
import sys
import tensorflow as tf
import numpy as np
import pathlib
import pandas as pd
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
    a2c_video_metadata = video_metadata[video_metadata["view"] == "a2c"]
    psax_video_metadata = video_metadata[video_metadata["view"] == "psax"]

    for view_metadata in (a4c_video_metadata, a2c_video_metadata, psax_video_metadata):
        view_metadata.reset_index(inplace=True, drop=True)

    train_ratio = 0.7
    validation_ratio = 0.15

    for view_metadata, view in zip((a4c_video_metadata, a2c_video_metadata, psax_video_metadata, video_metadata),
                                   ('a4c', 'a2c', 'psax', 'all')):
        train_folder = output_directory / 'tf_records' / view / 'train'
        test_folder = output_directory / 'tf_records' / view / 'test'
        validation_folder = output_directory / 'tf_records' / view / 'validation'

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

        width, height = echo_base.extract_metadata(train_samples.FileName[1], input_directory)

        print(f'Create train record for {view} echocardiograms.')
        number_of_train_samples = create_tf_record(input_directory, train_folder / 'train_{}.tfrecord', train_samples,
                                                   needed_frames)

        print(f'Create test record  for {view} echocardiograms.')
        number_of_test_samples = create_tf_record(input_directory, test_folder / 'test_{}.tfrecord', validation_samples,
                                                  needed_frames)

        print(f'Create validation record  for {view} echocardiograms')
        number_of_validation_samples = create_tf_record(input_directory, validation_folder / 'validation_{}.tfrecord',
                                                        test_samples, needed_frames)

        if not (output_directory / 'tf_records' / 'metadata.json').is_file():
            print('Calculate mean and standard deviation.')
            mean, std = echo_base.calculate_train_mean_and_std(input_directory, train_samples.FileName,
                                                               standardisation_sample)
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


if __name__ == '__main__':
    main(sys.argv[1:])
