from tensorflow.keras.datasets import mnist
import tensorflow as tf
import optuna

from projects.hyperparameter_optimisation.source.model import ImageClassifier

def load_data():
    data_split = mnist.load_data()
    return data_split

def reshape_data(img_array, img_height, img_width):
    return img_array.reshape((img_array.shape[0], img_height, img_width, 1))

def preprocess_data(data, num_classes, trgt_intnsty_min=0.0,
                    trgt_intnsty_max=1.0, src_intnsty_min=0, src_intnsty_max=255):
    data_x = data[0]
    data_y = data[1]

    data_x_scaled = ((trgt_intnsty_max - trgt_intnsty_min) *
            ((data_x - src_intnsty_min) / (src_intnsty_max - src_intnsty_min)) + trgt_intnsty_min)
    data_x_reshaped = reshape_data(data_x_scaled, img_height, img_width)
    data_y_transformed = tf.keras.utils.to_categorical(data_y, num_classes)

    return (data_x_reshaped, data_y_transformed)

def objective(trial, img_height, img_width, num_classes, training_data, evaluation_data):
    tf.keras.backend.clear_session()

    monitor = 'val_accuracy'
    objective_value_key = 'accuracy'

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    num_neurons = trial.suggest_categorical('num_neurons', [2, 4, 8, 16, 32])

    classifier = ImageClassifier()
    classifier.generate_model(img_height, img_width, num_classes, num_neurons)
    classifier.init_model(learning_rate)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3),
                 optuna.integration.KerasPruningCallback(trial, monitor)]
    _ = classifier.train(training_data, callbacks=callbacks)
    evaluation_result = classifier.evaluate(evaluation_data)

    return evaluation_result[objective_value_key]

if __name__ == "__main__":
    (training_data, evaluation_data) = load_data()
    img_height = training_data[0][0].shape[0]
    img_width = training_data[0][0].shape[1]
    num_classes = 10

    training_data_preprocessed = preprocess_data(training_data, num_classes)
    evaluation_data_preprocessed = preprocess_data(evaluation_data, num_classes)

    # vanilla training
    # classifier = ImageClassifier()
    # classifier.generate_model(img_height, img_width, num_classes)
    # classifier.init_model()
    #
    # history = classifier.train(training_data_preprocessed)
    # evaluation_result = classifier.evaluate(evaluation_data_preprocessed)
    # print('most recent accuracy on training data: {:.4f}'.format(history.history['accuracy'][-1]))
    # print('accuracy on evaluation data: {:.4f}'.format(evaluation_result['accuracy']))

    # hyperparameter optimisation using optuna
    # NOTE
    #   * the results of the study are depicted/visualised in the optuna dashboard
    #   * execute (in this example)
    #       optuna-dashboard sqlite:///example-study.db
    #     to open the dashboard

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
                                storage='sqlite:///data/example-study.db', study_name='example_study')
    study.optimize(lambda trial: objective(trial, img_height, img_width, num_classes, training_data_preprocessed,
                                           evaluation_data_preprocessed), n_trials=25, timeout=600)
    optimal_params = study.best_params
    print('optimal parameters:')
    print(' > learning_rate: {:.4f}'.format(optimal_params['learning_rate']))
    print(' > num_neurons: {:d}'.format(optimal_params['num_neurons']))

