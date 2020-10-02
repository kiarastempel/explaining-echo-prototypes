import tensorflow as tf
from tensorflow import keras


def test():
    # Load data
    test_paths = ['']
    test_set = tf.data.TFRecordDataset(test_paths, num_parallel_reads=8)
    model = keras.models.load_model('../saved/three_d_conv_best_model.h5')
    model.evaluate(test_set)


if __name__ == "__main__":
    # execute only if run as a script
    test()
