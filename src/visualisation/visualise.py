import io
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


def create_scatter_plot(x, y):
    sns.set_style('darkgrid')
    sns.scatterplot(x=x.numpy(), y=y.numpy())
    plt.plot([0, 100], [0, 100], 'black', lw=1)
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png')
    plt.close()
    plot_buf.seek(0)
    image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
    return tf.expand_dims(image, 0)
