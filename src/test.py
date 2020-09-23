from tensorflow import keras

def test():
    # Load data
    data = None
    model = keras.models.load_model('../saved/three_d_conv_best_model.h5')


if __name__ == "__main__":
    # execute only if run as a script
    test()
