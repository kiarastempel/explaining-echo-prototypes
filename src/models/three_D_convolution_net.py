from tensorflow import keras


class ThreeDConvolutionVGGStanford(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std):
        super(ThreeDConvolutionVGGStanford, self).__init__()

        input_shape = (frames, width, height, channels)
        self.model = keras.Sequential(
            [
                keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean/std,
                                                                  input_shape=input_shape),
                keras.layers.Conv3D(32, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                keras.layers.Conv3D(32, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                keras.layers.Conv3D(64, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv3D(64, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                keras.layers.Conv3D(128, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv3D(128, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv3D(128, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool3D(pool_size=(2, 1, 1)),
                keras.layers.Conv3D(256, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv3D(256, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Flatten(),

                keras.layers.Dense(2048, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs)


class ThreeDConvolutionResNetStanford(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std):
        super(ThreeDConvolutionResNetStanford, self).__init__()
        input_shape = (frames, width, height, channels)


    def call(self, inputs, **kwargs):
        return self.model(inputs)


class ThreeDConvolution(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std):
        super(ThreeDConvolution, self).__init__()
        input_shape = (frames, width, height, channels)
        self.model = keras.Sequential(
            [
                keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std),
                keras.layers.Conv3D(32, 3, activation='relu', input_shape=input_shape),
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
