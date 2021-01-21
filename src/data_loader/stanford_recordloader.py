import tensorflow as tf
import vidaug.augmentors as va


feature_description = {
    'frames': tf.io.VarLenFeature(dtype=tf.string),
    'ejection_fraction': tf.io.FixedLenFeature((), tf.float32),
    'number_of_frames': tf.io.FixedLenFeature((), tf.int64),
}


def build_dataset_validation(file_names):
    return build_dataset(file_names, 1, None, None, False)


def build_dataset(file_names, batch_size, shuffle_size, number_of_input_frames, augment=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_names)
    ds = ds.interleave(lambda files: tf.data.TFRecordDataset(files, compression_type='GZIP'), cycle_length=AUTOTUNE,
                       num_parallel_calls=AUTOTUNE)
    if shuffle_size is not None:
        ds = ds.shuffle(shuffle_size)
    if augment:
        ds = ds.map(lambda x: parse_and_augment_example(x, number_of_input_frames),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x: parse_example(x, number_of_input_frames),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    return ds.prefetch(AUTOTUNE)


def parse_example(example, number_of_input_frames):
    parsed_example = tf.io.parse_example(example, feature_description)
    raw_frames = tf.sparse.to_dense(parsed_example['frames'])
    number_of_frames = parsed_example['number_of_frames']
    if number_of_input_frames is None:
        number_of_input_frames = number_of_frames
    raw_subframes = raw_frames[0:number_of_input_frames]
    subframes = tf.map_fn(tf.io.decode_jpeg, raw_subframes, fn_output_signature=tf.uint8)
    subframes = tf.cast(subframes, tf.float32)
    return subframes, parsed_example['ejection_fraction']


def parse_and_augment_example(example, number_of_input_frames):
    parsed_example = tf.io.parse_example(example, feature_description)
    number_of_frames = parsed_example['number_of_frames']
    raw_frames = tf.sparse.to_dense(parsed_example['frames'])
    start = tf.cond(number_of_input_frames == number_of_frames,
                    lambda: tf.constant(0, dtype=tf.int64),
                    lambda: tf.random.uniform(shape=[], maxval=number_of_frames - number_of_input_frames,
                                              dtype=tf.int64))
    raw_subframes = raw_frames[start: start + number_of_input_frames]
    subframes = tf.map_fn(tf.io.decode_jpeg, raw_subframes, fn_output_signature=tf.uint8)
    subframes = tf.cast(subframes, tf.float32)
    augmented_subframes = tf.py_function(func=augment_test, inp=[(subframes,)], Tout=tf.float32)
    return augmented_subframes, parsed_example['ejection_fraction']


def augment_test(videos):
    seq = va.Sequential([
        va.RandomRotate(degrees=20),  # randomly rotates the video with a degree randomly chosen from [-10, 10]
    ])
    augmented_video = []
    for video in videos:
        augmented_video.append(seq(video.numpy()))

    return augmented_video
# translation
# rotation
# brightness