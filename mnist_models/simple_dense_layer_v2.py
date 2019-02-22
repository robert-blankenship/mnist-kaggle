import tensorflow


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
        model.add(tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax))
        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.categorical_crossentropy,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=epochs, batch_size=32,
                  validation_split=validation_split)

        self.model = model

