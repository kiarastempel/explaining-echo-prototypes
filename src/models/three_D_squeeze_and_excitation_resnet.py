from tensorflow import keras
import tensorflow as tf
from models.custom_layers import SqueezeAndExcitationResidualBlock, SqueezeExcitationResidualConvBlock


class ThreeDConvolutionSqueezeAndExciationResNet18(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output, regularization):
        super(ThreeDConvolutionSqueezeAndExciationResNet18, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape, dtype=tf.float32),
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=regularization),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 2 layer blocks
            SqueezeAndExcitationResidualBlock(64, 3, regularization),
            SqueezeAndExcitationResidualBlock(64, 3, regularization),

            SqueezeExcitationResidualConvBlock(128, 3,  regularization, 2),
            SqueezeAndExcitationResidualBlock(128, 3, regularization),

            SqueezeExcitationResidualConvBlock(256, 3, regularization, 2),
            SqueezeAndExcitationResidualBlock(256, 3, regularization),

            SqueezeExcitationResidualConvBlock(512, 3, regularization, 2),
            SqueezeAndExcitationResidualBlock(512, 3, regularization),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output, kernel_regularizer=regularization),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet34(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output , regularization):
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
            SqueezeAndExcitationResidualBlock(64, 3, regularization),
            SqueezeAndExcitationResidualBlock(64, 3, regularization),
            SqueezeAndExcitationResidualBlock(64, 3, regularization),

            SqueezeExcitationResidualConvBlock(128, 3, regularization, 2),
            SqueezeAndExcitationResidualBlock(128, 3, regularization),
            SqueezeAndExcitationResidualBlock(128, 3, regularization),
            SqueezeAndExcitationResidualBlock(128, 3, regularization),

            SqueezeExcitationResidualConvBlock(256, 3, regularization, 2),
            SqueezeAndExcitationResidualBlock(256, 3, regularization),
            SqueezeAndExcitationResidualBlock(256, 3, regularization),
            SqueezeAndExcitationResidualBlock(256, 3, regularization),
            SqueezeAndExcitationResidualBlock(256, 3, regularization),
            SqueezeAndExcitationResidualBlock(256, 3, regularization),

            SqueezeExcitationResidualConvBlock(512, 3, regularization, 2),
            SqueezeAndExcitationResidualBlock(512, 3, regularization),
            SqueezeAndExcitationResidualBlock(512, 3, regularization),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output, kernel_regularizer=regularization),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)
