import tensorflow


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
        model.add(tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax))
        model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                      loss=tensorflow.keras.losses.categorical_crossentropy,
                      metrics=[tensorflow.keras.metrics.categorical_accuracy])
        model.fit(mnist_data.images_train, mnist_data.labels_train, epochs=epochs, batch_size=32,
                  validation_split=validation_split)

        self.model = model
