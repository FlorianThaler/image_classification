import tensorflow_models as tfm


from source.hyperparameter_optimisation.main import load_data

if __name__ == '__main__':
    (training_data, evaluation_data) = load_data()

    augmentor = tfm.vision.

