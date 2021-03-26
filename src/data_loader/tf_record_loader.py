import tensorflow as tf
import vidaug.augmentors as va
from . import feature_descriptors
import random


def build_dataset(file_names, batch_size, shuffle_size, number_of_input_frames, resolution, augment=False,
                  dataset='stanford',
                  target='ejection_fraction',
                  full_video=False):

    feature_descriptor = feature_descriptors.mainz_feature_description if dataset == 'mainz' \
        else feature_descriptors.stanford_feature_description

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_names)
    ds = ds.interleave(lambda files: tf.data.TFRecordDataset(files, compression_type='GZIP'), cycle_length=AUTOTUNE,
                       num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x: parse_example(x, feature_descriptor, target, resolution), num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda video, y, number_of_frames: number_of_frames >= number_of_input_frames)
    if shuffle_size is not None:
        ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda video, y, number_of_frames: augment_example(video, y, number_of_frames,
                                                                       number_of_input_frames),
                    num_parallel_calls=AUTOTUNE)
    elif not full_video:
        ds = ds.map(lambda video, y, number_of_frames: first_frames(video, y, number_of_input_frames),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda video, y, number_of_frames: (tf.squeeze(video), y))
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    return ds.prefetch(AUTOTUNE)


@tf.autograph.experimental.do_not_convert
def parse_example(example, feature_descriptor, target, resolution):
    parsed_example = tf.io.parse_example(example, feature_descriptor)
    raw_frames = tf.sparse.to_dense(parsed_example['frames'])
    number_of_frames = parsed_example['number_of_frames']
    frames = tf.map_fn(tf.io.decode_jpeg, raw_frames, fn_output_signature=tf.uint8)

    if resolution is not None:
        frames = tf.image.resize_with_pad(frames, resolution[0], resolution[1])
    frames = tf.cast(frames, dtype=tf.float32)
    y = parsed_example['ejection_fraction', 'e_e_prime', ] if target == 'all' else parsed_example[target]
    return frames,  y, number_of_frames


@tf.autograph.experimental.do_not_convert
def augment_example(example, y, number_of_frames, number_of_input_frames):
    start = tf.cond(number_of_input_frames == number_of_frames,
                    lambda: tf.constant(0, dtype=tf.int64),
                    lambda: tf.random.uniform(shape=[], maxval=number_of_frames - number_of_input_frames,
                                              dtype=tf.int64))
    subframes = example[start: start + number_of_input_frames]
    augmented_subframes = tf.py_function(func=augmentation, inp=[(subframes,)], Tout=tf.float32)
    return augmented_subframes, y


@tf.autograph.experimental.do_not_convert
def first_frames(video, target, number_input_frames):
    video = tf.squeeze(video)
    return video[0: number_input_frames], target


def augmentation(videos):
    rare = lambda aug: va.Sometimes(0.01, aug)
    sometimes = lambda aug: va.Sometimes(0.02, aug)
    brightness_factor = random.uniform(0.9, 1.1)
    seq = va.Sequential([
        # rotation is very slow
        rare(va.RandomRotate(degrees=10)),
        va.RandomTranslate(10, 10),
        sometimes(va.Multiply(brightness_factor))
    ])
    augmented_video = []
    for video in videos:
        augmented_video.append(seq(video.numpy()))
    return augmented_video
