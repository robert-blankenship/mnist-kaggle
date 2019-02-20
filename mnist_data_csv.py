import numpy


class MnistDataCsv:
    # image_start_idx will be 0 for the training set, 1 for the Kaggle test set
    @staticmethod
    def get_images(csv_path, image_start_idx=1):
        with open(csv_path, "r") as csv:
            csv.readline()  # ignore the headers

            images_raw = []

            for line in csv:
                image = numpy.array(map(int, line.split(",")[image_start_idx:]))
                images_raw.append(image)

            images = numpy.array(images_raw, ndmin=2)
            images = numpy.absolute(images / 256.0 - .01)

            return images

    @staticmethod
    def get_images_2d(images):
        return

    @staticmethod
    def get_labels(csv_path):
        with open(csv_path, "r") as csv:
            csv.readline()  # ignore the headers

            labels_raw = []

            for line in csv:
                label = int(line.split(",")[0])
                labels_raw.append(label)

            num_images = len(labels_raw)

            labels = []

            for label in labels_raw:
                label_builder = numpy.zeros((10, 1))
                # zero-valued inputs can be problematic for neural networks, so use an arbitrary low value instead.
                label_builder = label_builder + 0.01
                label_builder[label] = 0.99
                labels.append(label_builder)

            labels = numpy.array(labels)
            labels = labels.reshape((num_images, 10))
            return labels

    def __init__(self, train_csv_path, test_csv_path):
        self.images_train = self.get_images(train_csv_path)
        self.images_train_2d = self.get_images_2d(self.images_train)
        self.labels_train = self.get_labels(train_csv_path)

        self.images_test = self.get_images(test_csv_path, image_start_idx=0)
