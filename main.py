# TODO: Reading all images at once seems to be pretty expensive.
import datetime
import logging
from mnist_data_csv import MnistDataCsv
import numpy
import tensorflow

logging.getLogger().setLevel(logging.DEBUG)


def generate_predictions_csv(model, images_test, output_csv_dir="submissions"):
    predictions = model.predict(images_test)

    with open("{}/{}.csv".format(output_csv_dir, datetime.datetime.now().isoformat()), "w") as output_csv:
        output_csv.write("ImageId,Label\n")

        for idx, prediction in enumerate(predictions):
            output_csv.write("{},{}\n".format(idx + 1, numpy.argmax(prediction)))

    print predictions


class MnistModelBasic:
    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, hidden_nodes=100, activation_function='sigmoid'):
        model = tensorflow.keras.Sequential()

        layers = [
            tensorflow.keras.layers.Dense(hidden_nodes, activation=activation_function,
                                          input_dim=mnist_data.images_train.shape[1]),
            tensorflow.keras.layers.Dense(10, activation=activation_function)
        ]

        for layer in layers:
            model.add(layer)

        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.mean_squared_error,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])

        logging.debug(mnist_data.images_train.shape)
        logging.debug(mnist_data.labels_train.shape)
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=10, batch_size=32, validation_split=0.2)

        self.model = model


class MnistModelConv2D:
    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, filters=3, kernel_size=3, activation_function='relu'):
        model = tensorflow.keras.Sequential()

        layers = [
            tensorflow.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_function,
                                           input_shape=(28, 28, 1), data_format="channels_last"),
            tensorflow.keras.layers.Conv2D(filters=3, kernel_size=5, strides=2, activation=activation_function),
            tensorflow.keras.layers.Conv2D(filters=3, kernel_size=5, strides=2, activation=activation_function),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(200, activation=activation_function),
            tensorflow.keras.layers.Dense(10, activation=activation_function)
        ]

        for layer in layers:
            model.add(layer)

        images_2d = []

        for image in mnist_data.images_train:
            # Grayscale, 28 by 28 pixel image with 1 channel.
            images_2d.append(numpy.roll(image, 28).reshape((28, 28, 1)))

        images_2d = numpy.array(images_2d, ndmin=4)

        numpy.set_printoptions(threshold=100)
        print images_2d[0].reshape((28, 28))

        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.mean_squared_error,  # categorical entropy thing doesn't work?
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])

        logging.debug(images_2d.shape)
        logging.debug(mnist_data.labels_train.shape)
        model.fit(images_2d, mnist_data.labels_train, epochs=10, batch_size=32, validation_split=0.2)

        self.model = model


def main():
    # Reading the csv is kind of slow!
    mnist_data = MnistDataCsv(train_csv_path="train.csv", test_csv_path="test.csv")

    # Validation Set Accuracy: 0.9752
    # model = MnistModelBasic(mnist_data, hidden_nodes=1000, activation_function='sigmoid')

    # Validation Set Accuracy: (only 0.70)
    model = MnistModelConv2D(mnist_data, filters=4, kernel_size=5, activation_function='relu')

    generate_predictions_csv(model.model, mnist_data.images_test)

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
