import tensorflow


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

