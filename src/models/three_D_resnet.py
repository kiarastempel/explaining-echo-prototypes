from tensorflow import keras
from models.custom_layers import ResidualBlock, ResidualBottleneckBlock, ResidualConvBlock, ResidualConvBottleneckBlock


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

            # variable 2 layer blocks
            ResidualBlock(64, 3),
            ResidualBlock(64, 3),

            ResidualConvBlock(3, 128),
            ResidualBlock(3, 128),

            ResidualConvBlock(3, 256),
            ResidualBlock(3, 256),

            ResidualConvBlock(3, 512),
            ResidualBlock(3, 512),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet34(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionResNet34, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 2 layer blocks
            ResidualBlock(64, 3),
            ResidualBlock(64, 3),
            ResidualBlock(64, 3),

            ResidualConvBlock(3, 128),
            ResidualBlock(3, 128),
            ResidualBlock(3, 128),
            ResidualBlock(3, 128),

            ResidualConvBlock(3, 256),
            ResidualBlock(3, 256),
            ResidualBlock(3, 256),
            ResidualBlock(3, 256),
            ResidualBlock(3, 256),
            ResidualBlock(3, 256),

            ResidualConvBlock(3, 512),
            ResidualBlock(3, 512),
            ResidualBlock(3, 512),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet50(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionResNet50, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 3 layer blocks
            ResidualConvBottleneckBlock(3, 64),
            ResidualBottleneckBlock(3, 64),
            ResidualBottleneckBlock(3, 64),

            ResidualConvBottleneckBlock(3, 128),
            ResidualBottleneckBlock(3, 128),
            ResidualBottleneckBlock(3, 128),
            ResidualBottleneckBlock(3, 128),

            ResidualConvBottleneckBlock(3, 256),
            ResidualBottleneckBlock(3, 256),
            ResidualBottleneckBlock(3, 256),
            ResidualBottleneckBlock(3, 256),
            ResidualBottleneckBlock(3, 256),
            ResidualBottleneckBlock(3, 256),

            ResidualConvBottleneckBlock(3, 512),
            ResidualBottleneckBlock(3, 512),
            ResidualBottleneckBlock(3, 512),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)