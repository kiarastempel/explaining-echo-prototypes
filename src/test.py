from pathlib import Path
from tensorflow import keras

from data_loader import record_loader


def test():
    data_folder = Path('../data/dynamic-echo-data/tf_record/')
    test_record_file_name = data_folder / 'test' / 'test_*.tfrecord'
    test_set = record_loader.build_dataset(str(test_record_file_name))
    model_path = '../saved/three_d_conv_best_model.h5'
    model = keras.models.load_model(model_path)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_set, batch_size=128)
    print("test loss, test mse:", results)


if __name__ == "__main__":
    # execute only if run as a script
    test()
