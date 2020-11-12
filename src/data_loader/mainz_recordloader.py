import tensorflow as tf
from tensorflow import keras


feature_description = {
    'frames': tf.io.VarLenFeature(dtype=tf.string),
    'ejection_fraction': tf.io.FixedLenFeature((), tf.float32),
    'number_of_frames': tf.io.FixedLenFeature((), tf.int64),
    'e_e_prime': tf.io.FixedLenFeature((), tf.float32),
    'quality': tf.io.FixedLenFeature((), tf.int64)
}
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomContrast(0.2),
        keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        keras.layers.experimental.preprocessing.RandomRotation(0.5)
    ])


def build_dataset(file_names, batch_size, shuffle_size, number_of_input_frames, augment=False,
                  split=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset \
        .list_files(file_names) \
        .interleave(tf.data.TFRecordDataset, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE) \
        .shuffle(shuffle_size)
    if split:
        ds = ds.map(lambda x: parse_and_augment_example(x, number_of_input_frames),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x: parse_example(x, number_of_input_frames),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    # if augment:
    #   ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def parse_example( example, number_of_input_frames):
    parsed_example = tf.io.parse_example(example, feature_description)
    raw_frames = tf.sparse.to_dense(parsed_example['frames'])
    number_of_frames = parsed_example['number_of_frames']
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

    return subframes, parsed_example['ejection_fraction']