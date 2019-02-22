# TODO: Reading all images at once seems to be pretty expensive.
import datetime
import logging
from mnist_data_csv import MnistDataCsv
import mnist_models.conv_2d_v1
import numpy
import os

logging.getLogger().setLevel(logging.DEBUG)


def generate_predictions_csv(model, images_test, output_csv_dir="submissions", uses_2d_images=False):
    if uses_2d_images:
        predictions = model.predict(mnist_models.conv_2d_v1.MnistModelConv2D.convert_images(images_test))
    else:
        predictions = model.predict(images_test)

    if not os.path.exists(output_csv_dir):
        os.mkdir(output_csv_dir)

    with open("{}/{}.csv".format(output_csv_dir, datetime.datetime.now().isoformat()), "w") as output_csv:
        output_csv.write("ImageId,Label\n")

        for idx, prediction in enumerate(predictions):
            output_csv.write("{},{}\n".format(idx + 1, numpy.argmax(prediction)))

    print predictions


def main():
    # TODO: Improve performance of reading the CSV.
    # Reading the csv is kind of slow!
    mnist_data = MnistDataCsv(train_csv_path="data/train.csv", test_csv_path="data/test.csv")

    # model = MnistModelBasic(mnist_data, epochs=10)
    # model = MnistModelBasic2(mnist_data)
    # model = MnistModelBasic3(mnist_data)
    # model = MnistModelBasic4(mnist_data)
    # generate_predictions_csv(model.model, mnist_data.images_test)

    model = mnist_models.conv_2d_v1.MnistModelConv2D(mnist_data)
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
