import tensorflow as tf
from tensorflow import keras


def build_dataset(file_names, batch_size, shuffle_size, number_of_input_frames, augment=False, split=False):
    data_augmentation = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.RandomContrast(0.2),
            keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
            keras.layers.experimental.preprocessing.RandomRotation(0.5)
        ]
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset \
        .list_files(file_names) \
        .interleave(tf.data.TFRecordDataset, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE) \
        .shuffle(shuffle_size)
    if split:
        ds = ds.map(lambda x: parse_and_augment_example(x, number_of_input_frames), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x: parse_example(x, number_of_input_frames), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


feature_description = {
        'frames': tf.io.VarLenFeature(dtype=tf.string),
        'ejection_fraction': tf.io.FixedLenFeature((), tf.float32),
        'number_of_frames': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
    }


def parse_example(example, number_of_input_frames):
    parsed_example = tf.io.parse_example(example, feature_description)
    width = parsed_example['width']
    height = parsed_example['height']
    channels = 1
    raw_frames = tf.sparse.to_dense(parsed_example['frames'])
    number_of_frames = parsed_example['number_of_frames']
    frames = tf.io.decode_raw(raw_frames, out_type=tf.uint8)
    frames = frames[0:number_of_input_frames]
    frames = tf.cast(frames, tf.float32)
    frames = tf.reshape(frames, (number_of_input_frames, width, height, channels))
    return frames, parsed_example['ejection_fraction']


def parse_and_augment_example(example, number_of_input_frames):

    parsed_example = tf.io.parse_example(example, feature_description)
    width = parsed_example['width']
    height = parsed_example['height']
    channels = 1
    number_of_frames = parsed_example['number_of_frames']
    raw_frames = tf.sparse.to_dense(parsed_example['frames'])
    frames = tf.io.decode_raw(raw_frames, out_type=tf.uint8)
    frames = tf.cast(frames, tf.float32)
    if number_of_input_frames == number_of_frames:
        start = 0
    else:
        start = tf.random.uniform(shape=[], maxval=number_of_frames - number_of_input_frames, dtype=tf.int64)
    subframes = frames[start: start + number_of_input_frames]
    subframes = tf.reshape(subframes, (number_of_input_frames, width, height, channels))
    return subframes, parsed_example['ejection_fraction']
