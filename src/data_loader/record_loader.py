import tensorflow as tf
from tensorflow import keras


def build_dataset(file_names, batch_size, shuffle_size, augment=False):
    data_augmentation = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.RandomContrast(0.2),
            keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
            keras.layers.experimental.preprocessing.RandomRotation(0.5)
        ]
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset\
        .list_files(file_names)\
        .interleave(tf.data.TFRecordDataset, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE)\
        .shuffle(shuffle_size)\
        .batch(batch_size=batch_size, drop_remainder=True)\
        .map(map_func=parse_examples_batch, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def parse_examples_batch(examples):
    feature_description = {
        'frames': tf.io.FixedLenFeature((50), dtype=tf.string),
        'ejection_fraction': tf.io.FixedLenFeature((), tf.float32),
        'fps': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
    }
    batch = len(examples)
    parsed_examples = tf.io.parse_example(examples, feature_description)
    width = parsed_examples['width']
    height = parsed_examples['height']
    number_of_frames = parsed_examples['fps']
    channels = 1
    frames = tf.io.decode_raw(parsed_examples['frames'], out_type=tf.uint8)
    frames = tf.cast(frames, tf.float32)
    frames = tf.reshape(frames, (batch, number_of_frames[0], width[0], height[0], channels))
    return frames, parsed_examples['ejection_fraction']
