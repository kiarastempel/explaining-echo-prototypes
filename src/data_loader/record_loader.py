import tensorflow as tf


def build_dataset(file_names):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    return tf.data.Dataset.list_files(
        file_names
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE
    ).shuffle(
        2048
    ).batch(
        batch_size=64,
        drop_remainder=True,
    ).map(
        map_func=parse_examples_batch,
        num_parallel_calls=AUTOTUNE
    ).cache(
    ).prefetch(AUTOTUNE
               )


def parse_examples_batch(examples):
    feature_description = {
        'frames': tf.io.FixedLenFeature((50), dtype=tf.string),
        'ejection_fraction': tf.io.FixedLenFeature((), tf.float32),
        'fps': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
    }

    parsed_examples = tf.io.parse_example(examples, feature_description)
    width = parsed_examples['width']
    height = parsed_examples['height']
    number_of_frames = parsed_examples['fps']
    batch = 64
    channels = 1
    frames = tf.io.decode_raw(parsed_examples['frames'], out_type=tf.uint8)
    frames = tf.cast(frames, tf.float32)
    frames = tf.reshape(frames, (batch, number_of_frames[0], width[0], height[0], channels))
    return frames, parsed_examples['ejection_fraction']
