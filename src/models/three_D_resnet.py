from tensorflow import keras
from models.custom_layers import ResidualBlock, ResidualBottleneckBlock, ResidualConvBlock, ResidualConvBottleneckBlock


class ThreeDConvolutionResNet18(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output, regularization):
        super(ThreeDConvolutionResNet18, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=regularization),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 2 layer blocks
            ResidualBlock(64, 3, regularization),
            ResidualBlock(64, 3, regularization),

            ResidualConvBlock(128, 3, regularization, 2),
            ResidualBlock(128, 3, regularization),

            ResidualConvBlock(256, 3, regularization, 2),
            ResidualBlock(256, 3, regularization),

            ResidualConvBlock(512, 3, regularization, 2),
            ResidualBlock(512, 3, regularization),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output, kernel_regularizer=regularization),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet34(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output, regularization):
        super(ThreeDConvolutionResNet34, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=regularization),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 2 layer blocks
            ResidualBlock(64, 3, regularization),
            ResidualBlock(64, 3, regularization),
            ResidualBlock(64, 3, regularization),

            ResidualConvBlock(128, 3, 2),
            ResidualBlock(128, 3, regularization),
            ResidualBlock(128, 3, regularization),
            ResidualBlock(128, 3, regularization),

            ResidualConvBlock(256, regularization, 2),
            ResidualBlock(256, 3, regularization),
            ResidualBlock(256, 3, regularization),
            ResidualBlock(256, 3, regularization),
            ResidualBlock(256, 3, regularization),
            ResidualBlock(256, 3, regularization),

            ResidualConvBlock(512, 3, regularization, 2),
            ResidualBlock(512, 3, regularization),
            ResidualBlock(512, 3, regularization),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output, kernel_regularizer=regularization),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet50(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output, regularization):
        super(ThreeDConvolutionResNet50, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=regularization),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 3 layer blocks
            ResidualConvBottleneckBlock(64, 3, 2),
            ResidualBottleneckBlock(64, 3, regularization),
            ResidualBottleneckBlock(64, 3, regularization),

            ResidualConvBottleneckBlock(128, 3, regularization, 2),
            ResidualBottleneckBlock(128, 3, regularization),
            ResidualBottleneckBlock(128, 3, regularization),
            ResidualBottleneckBlock(128, 3, regularization),

            ResidualConvBottleneckBlock(256, 3, regularization, 2),
            ResidualBottleneckBlock(256, 3, regularization),
            ResidualBottleneckBlock(256, 3, regularization),
            ResidualBottleneckBlock(256, 3, regularization),
            ResidualBottleneckBlock(256, 3, regularization),
            ResidualBottleneckBlock(256, 3, regularization),

            ResidualConvBottleneckBlock(512, 3, 2),
            ResidualBottleneckBlock(512, 3, regularization),
            ResidualBottleneckBlock(512, 3, regularization),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)
