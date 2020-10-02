import tensorflow as tf


def parse_example_batch(examples):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'ef': tf.io.FixedLenFeature([], tf.float32, default_value=0)
    }
    parsed_examples = tf.io.parse_example(examples, feature_description)
    images = []
    for i in range(len(parsed_examples)):
        images.append(tf.io.decode_image(parsed_examples['image'][i], channels=1))
    return images, parsed_examples['ef']


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
        map_func=parse_example_batch,
        num_parallel_calls=AUTOTUNE
    ).cache(
    ).prefetch(AUTOTUNE
               )
