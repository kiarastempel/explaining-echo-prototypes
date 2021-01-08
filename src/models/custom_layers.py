from tensorflow import keras
import tensorflow as tf


class CustomConv3D(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, strides=1, use_bn=False, padding='valid', **kwargs):
        super(CustomConv3D, self).__init__(**kwargs)
        self.custom_conv_3d = keras.Sequential()
        self.custom_conv_3d.add(keras.layers.Conv3D(kernel_number, kernel_size, strides))
        if use_bn:
            self.custom_conv_3d.add(keras.layers.BatchNormalization())
        self.custom_conv_3d.add(keras.layers.ReLU())

    def call(self, inputs, training=None):
        return self.custom_conv_3d(inputs)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.resnet_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, kernel_size, 1, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number, kernel_size, padding='same'),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_block(inputs)
        output_sum = tf.add(intermediate_output, inputs)
        return self.relu(output_sum)


class ResidualBottleneckBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, **kwargs):
        super(ResidualBottleneckBlock, self).__init__(**kwargs)
        self.resnet_bottleneck_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, use_bn=True),
                CustomConv3D(kernel_number, kernel_size, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number * 4, 1),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        intermediate_output = self.resnet_bottleneck_block(inputs)
        output_sum = tf.add(intermediate_output, inputs)
        return self.relu(output_sum)


class ResidualConvBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.resnet_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, kernel_size, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number, kernel_size, padding='same'),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_bottleneck_block(inputs)
        shortcut = self.shortcut_conv(inputs)
        output_sum = tf.add(intermediate_output, shortcut)
        return self.relu(output_sum)


class ResidualConvBottleneckBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, **kwargs):
        super(ResidualConvBottleneckBlock, self).__init__(**kwargs)
        self.resnet_bottleneck_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, 1, use_bn=True),
                CustomConv3D(kernel_number, kernel_size, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number * 4, 1),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()
        self.shortcut_conv = keras.layers.Conv3D(kernel_number * 4, 1)

    def call(self, inputs, **kwargs):
        intermediate_output = self.resnet_bottleneck_block(inputs)
        shortcut = self.shortcut_conv(inputs)
        output_sum = tf.add(intermediate_output, shortcut)
        return self.relu(output_sum)


class SqueezeAndExcitationPath(keras.layers.Layer):
    def __init__(self, channel, ratio=16, **kwargs):
        super(SqueezeAndExcitationPath, self).__init__(**kwargs)
        self.se_path = keras.Sequential(
            [
                keras.layers.GlobalAvgPool3D(),
                keras.layers.Dense(int(channel/ratio), activation='relu'),
                keras.layers.Dense(channel, activation='sigmoid'),
            ]
        )

    def call(self, inputs, **kwargs):
        return self.se_path(inputs)
