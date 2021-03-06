import datetime
import logging
from mnist_data_csv import MnistDataCsv
import mnist_models.conv_2d_v2
# import mnist_models.simple_dense_layer_v1
import numpy
import os

logging.getLogger().setLevel(logging.DEBUG)


def generate_predictions_csv(mnist_model, mnist_data, output_csv_dir="submissions"):
    if getattr(mnist_model, 'uses_2d_images', False):
        predictions = mnist_model.model.predict(mnist_data.images_test_2d)
    else:
        predictions = mnist_model.model.predict(mnist_data.images_test)

    if not os.path.exists(output_csv_dir):
        os.mkdir(output_csv_dir)

    with open("{}/{}.csv".format(output_csv_dir, datetime.datetime.now().isoformat()), "w") as output_csv:
        output_csv.write("ImageId,Label\n")

        for idx, prediction in enumerate(predictions):
            output_csv.write("{},{}\n".format(idx + 1, numpy.argmax(prediction)))


def main():
    # TODO: Improve performance of reading the CSV.
    # Reading the csv is kind of slow!
    mnist_data = MnistDataCsv(train_csv_path="data/train.csv", test_csv_path="data/test.csv")

    # model = mnist_models.simple_dense_layer_v1.MnistModelBasic(mnist_data, epochs=1)
    # model = MnistModelBasic2(mnist_data)
    # model = MnistModelBasic3(mnist_data)
    # model = MnistModelBasic4(mnist_data)

    # model = mnist_models.conv_2d_v1.MnistModelConv2D(mnist_data)
    model = mnist_models.conv_2d_v2.MnistModelConv2D_v2(mnist_data)
    generate_predictions_csv(model, mnist_data)

    print model


if __name__ == '__main__':
    main()
