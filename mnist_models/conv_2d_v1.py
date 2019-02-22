import numpy
import tensorflow


# Amazing performance, and very low variance!
# training: 99.64%, dev: 99.31% (dev set performance within .1% difference between runs).
# submissions/2019-02-21T13:24:36.717835.csv
# Kaggle: 99.214%
# TODO:
#  - Try more epochs
#  - Try adding regularization to the Dense neural layers
#  - Are there regularization techniques for the Conv2D layers.
#  - Try data augmentation?
class MnistModelConv2D:
    """
    :type mnist_data: MnistDataCsv
    """
    uses_2d_images = True

    def __init__(self, mnist_data, filters=5, kernel_size=3, activation_function=tensorflow.nn.leaky_relu, validation_split=0.1):
        model = tensorflow.keras.Sequential()

        layers = [
            tensorflow.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_function,
                                           input_shape=(28, 28, 1), data_format="channels_last"),
            tensorflow.keras.layers.Conv2D(filters=10, kernel_size=3, strides=1, activation=activation_function),
            tensorflow.keras.layers.MaxPool2D(),
            tensorflow.keras.layers.Conv2D(filters=20, kernel_size=3, strides=1, activation=activation_function),
            tensorflow.keras.layers.MaxPool2D(),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(100, activation=activation_function),
            tensorflow.keras.layers.Dense(10, activation=activation_function)
        ]

        for layer in layers:
            model.add(layer)

        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.mean_squared_error,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(self.convert_images(mnist_data.images_train), mnist_data.labels_train, epochs=10,
                  batch_size=32, validation_split=validation_split)
        self.model = model

    @staticmethod
    def convert_images(images):
        images_2d = []

        for image in images:
            # Grayscale, 28 by 28 pixel image with 1 channel.
            images_2d.append(numpy.roll(image, 28).reshape((28, 28, 1)))

        # numpy.set_printoptions(threshold=100)
        # print images_2d[0].reshape((28, 28))

        return numpy.array(images_2d, ndmin=4)


