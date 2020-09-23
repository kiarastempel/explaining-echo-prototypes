from tensorflow import keras


class ThreeDConvolution(keras.Model):
    def __init__(self):
        super(ThreeDConvolution, self).__init__()
        self.model = keras.Sequential(
            [
                keras.layers.Conv3D(32, 3, activation='relu'),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                keras.layers.Conv3D(32, 3, activation='relu'),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                keras.layers.Conv3D(64, 3, activation='relu'),
                keras.layers.Conv3D(64, 3, activation='relu'),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                keras.layers.Conv3D(128, 3, activation='relu'),
                keras.layers.Conv3D(128, 3, activation='relu'),
                keras.layers.Conv3D(128, 3, activation='relu'),
                keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
                keras.layers.Conv3D(256, 3, activation='relu'),
                keras.layers.Conv3D(256, 3, activation='relu'),
                keras.layers.Conv3D(256, 3, activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs)
