from datetime import datetime
from tensorflow import keras
from models import three_D_convolution_net
import tensorflow_addons as tfa
import tensorflow as tf

EPOCHS = 200


def train():
    model = three_D_convolution_net.ThreeDConvolution()
    opt = keras.optimizers.Adam(0.001)
    opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    model.compile(
        optimizer=opt,
        loss=keras.losses.MSE(),
        metrics=keras.metrics.MSE()
    )
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20),
        keras.callbacks.ModelCheckpoint(filepath='../saved/three_d_conv_best_model.h5', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    echo_dataset = tf.data.TFRecordDataset()

    x_train = None
    x_val = None
    y_train = None
    y_val = None
    batch_size = None
    model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, callbacks=callbacks)


if __name__ == "__main__":
    # execute only if run as a script
    train()
