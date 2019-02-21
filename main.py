# TODO: Reading all images at once seems to be pretty expensive.
import datetime
import logging
from mnist_data_csv import MnistDataCsv
import numpy
import os
import tensorflow

logging.getLogger().setLevel(logging.DEBUG)


def generate_predictions_csv(model, images_test, output_csv_dir="submissions", uses_2d_images=False):
    if uses_2d_images:
        predictions = model.predict(MnistModelConv2D.convert_images(images_test))
    else:
        predictions = model.predict(images_test)

    if not os.path.exists(output_csv_dir):
        os.mkdir(output_csv_dir)

    with open("{}/{}.csv".format(output_csv_dir, datetime.datetime.now().isoformat()), "w") as output_csv:
        output_csv.write("ImageId,Label\n")

        for idx, prediction in enumerate(predictions):
            output_csv.write("{},{}\n".format(idx + 1, numpy.argmax(prediction)))

    print predictions

# This is my best and simplest neural network so far.
# 99.5% training, 97.9% dev
class MnistModelBasic:
    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, validation_split=0.2, epochs=10, activation_function=tensorflow.nn.leaky_relu):
        model = tensorflow.keras.Sequential()
        model.add(tensorflow.keras.layers.Dense(400, activation=activation_function,
                                                input_dim=mnist_data.images_train[0].shape[0]))
        model.add(tensorflow.keras.layers.Dense(150, activation=activation_function)),
        model.add(tensorflow.keras.layers.Dense(50, activation=activation_function)),
        model.add(tensorflow.keras.layers.Dense(10, activation=activation_function))
        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.mean_squared_error,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=epochs, batch_size=32,
                  validation_split=validation_split)

        self.model = model

# Around 99.8% training, ~98% dev
# submissions/2019-02-21T12:08:01.925978.csv
class MnistModelBasic2:
    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, validation_split=0.2, epochs=100, activation_function=tensorflow.nn.leaky_relu):
        model = tensorflow.keras.Sequential()
        model.add(tensorflow.keras.layers.Dense(500, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.2))
        model.add(tensorflow.keras.layers.Dense(250, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.2))
        model.add(tensorflow.keras.layers.Dense(125, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.2))
        model.add(tensorflow.keras.layers.Dense(75, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.2))
        model.add(tensorflow.keras.layers.Dense(30, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.2))
        model.add(tensorflow.keras.layers.Dense(10, activation=activation_function))
        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.mean_squared_error,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=epochs, batch_size=32,
                  validation_split=validation_split)

        self.model = model

# training: 1.00%, dev: 98.0% (dev set performance can be quite noisy).
# submissions/2019-02-21T12:25:16.157163.csv
# kaggle test set: 98.057%
class MnistModelBasic3:
    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, validation_split=0.1, epochs=100, activation_function=tensorflow.nn.leaky_relu):
        model = tensorflow.keras.Sequential()
        model.add(tensorflow.keras.layers.Dense(500, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.33))
        model.add(tensorflow.keras.layers.Dense(200, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.33))
        model.add(tensorflow.keras.layers.Dense(10, activation=activation_function))
        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.mean_squared_error,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=epochs, batch_size=32,
                  validation_split=validation_split)

        self.model = model

class MnistModelBasic4:
    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, validation_split=0.1, epochs=10, activation_function=tensorflow.nn.leaky_relu):
        self.uses_softmax = True

        model = tensorflow.keras.Sequential()
        model.add(tensorflow.keras.layers.Dense(500, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.33))
        model.add(tensorflow.keras.layers.Dense(200, activation=activation_function)),
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Dropout(0.5))
        model.add(tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax))
        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.categorical_crossentropy,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=epochs, batch_size=32,
                  validation_split=validation_split)

        self.model = model

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


def main():
    # Reading the csv is kind of slow!
    mnist_data = MnistDataCsv(train_csv_path="train.csv", test_csv_path="test.csv")

    # model = MnistModelBasic(mnist_data, epochs=10)
    # model = MnistModelBasic2(mnist_data)
    # model = MnistModelBasic3(mnist_data)
    # model = MnistModelBasic4(mnist_data)
    # generate_predictions_csv(model.model, mnist_data.images_test)

    model = MnistModelConv2D(mnist_data)
    generate_predictions_csv(model.model, mnist_data.images_test, uses_2d_images=model.uses_2d_images)

    print model


if __name__ == '__main__':
    main()


    # for i in range(0, 100):
    #     label = numpy.argmax(mnist_data.labels[i])
    #     actual = network.predict(mnist_data.images[i:i+1])
    #     actual_class = numpy.argmax(actual)
    #     logging.debug("prediction={}, actual={}".format(label, actual_class))
    #     logging.debug(mnist_data.labels[i])
    #     logging.debug(actual)
