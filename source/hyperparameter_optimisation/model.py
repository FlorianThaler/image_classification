import tensorflow as tf

class ImageClassifier:
    def __init__(self):
        self._model = tf.keras.Model()

    def generate_model(self, img_width, img_height, num_classes, num_neurons=10):
        x = tf.keras.Input(shape=(img_width, img_height, 1))
        y = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu)(y)
        y = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(y)
        self._model = tf.keras.Model(inputs=x, outputs=y)

    def init_model(self, learning_rate=1e-4):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, training_data, num_epochs=3, callbacks=None):
        training_data_x = training_data[0]
        training_data_y = training_data[1]
        if callbacks:
            history = self._model.fit(training_data_x, training_data_y, epochs=num_epochs, callbacks=callbacks)
        else:
            history = self._model.fit(training_data_x, training_data_y, epochs=num_epochs)

        return history

    def evaluate(self, evaluation_data):
        evaluation_data_x = evaluation_data[0]
        evaluation_data_y = evaluation_data[1]

        evaluation_result = self._model.evaluate(evaluation_data_x, evaluation_data_y, return_dict=True)
        return evaluation_result
