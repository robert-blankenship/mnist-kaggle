import tensorflow


# I never got the chance to run this one. The notable difference is that there is a pretty high dropout factor.
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

