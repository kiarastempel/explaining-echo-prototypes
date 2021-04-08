import io
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def create_scatter_plot(x, y, target, dataset):
    sns.set_style('darkgrid')
    sns.scatterplot(x=x, y=y)
    plt.plot([0, np.max(x)], [0, np.max(y)], 'black', lw=1)
    plt.xlabel('Manual measured values')
    plt.ylabel('Automatic predicted values')
    plt.title(f'Comparison of {target} on {dataset} data')
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png')
    plot_buf.seek(0)
    image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
    plt.clf()
    return tf.expand_dims(image, 0)
