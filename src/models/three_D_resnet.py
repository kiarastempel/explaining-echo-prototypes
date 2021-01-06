from tensorflow import keras
from custom_layers import ResidualBlock, CustomConv3D, ResidualBottleneckBlock


class ThreeDConvolutionResNet18(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionResNet18, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable blocks
            ResidualBlock(),
            ResidualBlock(),

            ResidualBlock(),
            ResidualBlock(),

            ResidualBlock(),
            ResidualBlock(),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)
