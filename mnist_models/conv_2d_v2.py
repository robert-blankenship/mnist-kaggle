import tensorflow


# Amazing performance, and very low variance!
# training: 99.98%, dev: 100.0% (dev set performance within .02% difference between runs).
# submissions/2019-02-22T06:50:25.171165.csv
# TODO:
#  - Try adding regularization to the Dense neural layers
#  - Are there regularization techniques for the Conv2D layers.
#  - Try data augmentation?
class MnistModelConv2D_v2:
    uses_2d_images = True

    """
    :type mnist_data: MnistDataCsv
    """
    def __init__(self, mnist_data, filters=5, kernel_size=3, activation_function=tensorflow.nn.relu,
                 validation_split=0.1):
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

        model.compile(optimizer=tensorflow.train.AdamOptimizer(.01),
                      loss=tensorflow.keras.losses.mean_squared_error,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model = tensorflow.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tensorflow.contrib.tpu.TPUDistributionStrategy(
                tensorflow.contrib.cluster_resolver.TPUClusterResolver("node-1")))

        # May need to use different labels for softmax
        model.fit(mnist_data.images_train_2d, mnist_data.labels_train, epochs=50,
                  batch_size=512, validation_split=validation_split)
        self.model = model
