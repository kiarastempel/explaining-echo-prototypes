from tensorflow import keras
from models.custom_layers import CustomConv3D


class ThreeDConvolutionVGGStanford(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, batch_normalization=False):
        super(ThreeDConvolutionVGGStanford, self).__init__()

        input_shape = (frames, width, height, channels)
        self.model = keras.Sequential(
            [
                keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                                  input_shape=input_shape),
                CustomConv3D(32, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                CustomConv3D(32, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                CustomConv3D(64, 3, use_bn=batch_normalization),
                CustomConv3D(64, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                CustomConv3D(128, 3, use_bn=batch_normalization),
                CustomConv3D(128, 3, use_bn=batch_normalization),
                CustomConv3D(128, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(2, 1, 1)),
                CustomConv3D(256, 3, use_bn=batch_normalization),
                CustomConv3D(256, 3, use_bn=batch_normalization),
                keras.layers.Flatten(),

                keras.layers.Dense(2048),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU,
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(1)
            ]
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs)




class ThreeDConvolutionVVG(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, batch_normalization=True):
        super(ThreeDConvolutionVVG, self).__init__()
        input_shape = (frames, width, height, channels)
        self.model = keras.Sequential(
            [
                keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                                  input_shape=input_shape),
                CustomConv3D(32, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                CustomConv3D(32, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                CustomConv3D(64, 3, use_bn=batch_normalization),
                CustomConv3D(64, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(1, 2, 2)),
                CustomConv3D(128, 3, use_bn=batch_normalization),
                CustomConv3D(128, 3, use_bn=batch_normalization),
                CustomConv3D(128, 3, use_bn=batch_normalization),
                keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
                CustomConv3D(256, 3, use_bn=batch_normalization),
                CustomConv3D(256, 3, use_bn=batch_normalization),
                CustomConv3D(256, 3, use_bn=batch_normalization),
                keras.layers.Flatten(),
                keras.layers.Dense(2048),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs)


