# Project
import sample_generator
import auxiliary_functions
import plotting

# Additional
from cntk import Trainer, cntk_device, StreamConfiguration, learning_rate_schedule, UnitType
from cntk.learner import sgd
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss

import numpy as np


class Model:
    """The model used for training."""
    """ (static) fields """
    # info: static fields are shared among all instantiations of a class, e.g.:
    # Model.modelCount
    # [out]: 0
    # firstModel = Model(parameters, 2, 2)
    # Model.modelCount
    # [out]: 1
    # secondModel = Model(parameters, 3, 3)
    # Model.modelCount
    # [out]: 2

    # Counts how many model have been instantiated
    model_count = 0

    """ constructor """

    def __init__(self, model_parameters: '{string : parameter}', input_dimension: int, output_dimension: int):
        """Initializes a new instance of the Model class."""
        self.model_parameters = model_parameters
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input = input_variable
        self.label = input_variable
        self.trainer = Trainer

        Model.model_count += 1

    """ methods """

    def linear_layer(self) \
            -> object:
        """Creates the linear layer function object (?)."""
        self.input = input_variable(self.input_dimension, np.float32)
        weight_parameter = parameter(shape = (self.input_dimension, self.output_dimension))
        bias_parameter = parameter(shape = (self.output_dimension))

        self.model_parameters['w'], self.model_parameters['b'] = weight_parameter, bias_parameter

        return times(self.input, weight_parameter) + bias_parameter

    def initialize_trainer(self, z: object, output_dimension: int):
        """Initializes the trainer object."""
        self.label = input_variable((output_dimension), np.float32)
        loss = cross_entropy_with_softmax(z, self.label)
        eval_error = classification_error(z, self.label)

        # Instantiate the trainer object to drive the model training
        learning_rate = 0.5
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        learner = sgd(z.parameters, lr_schedule)
        self.trainer = Trainer(z, loss, eval_error, [learner])

    def train(self, minibatch_size: int, num_samples_to_train: int)\
            -> '{string: [int]}':
        """Trains the model using the trainer."""
        num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

        training_progress_output_frequency = 50

        plot_data = {"batchsize": [], "loss": [], "error": []}

        for i in range(0, num_minibatches_to_train):
            features, labels = sample_generator.generate_random_data_sample(
                minibatch_size,
                self.input_dimension,
                self.output_dimension)

            # Specify input variables mapping in the model to actual minibatch data to be trained with
            self.trainer.train_minibatch({self.input: features, self.label: labels})
            batch_size, loss, error = Model.print_training_progress(
                self.trainer,
                i,
                training_progress_output_frequency,
                verbose=1)

            if not (loss == "NA" or error == "NA"):
                plot_data["batchsize"].append(batch_size)
                plot_data["loss"].append(loss)
                plot_data["error"].append(error)

        # Compute the moving average to smooth out the loss in SGD
        # By indexing the dictionary with keys which are not yet supplied (e.g. "avgloss"), they will automatically
        # be added to the dictionary
        plot_data["avgloss"] = auxiliary_functions.moving_average(plot_data["loss"])
        plot_data["avgerror"] = auxiliary_functions.moving_average(plot_data["error"])

        return plot_data

    def test_trained_model(self, z: object, test_minibatch_size: int):
        """Tests the trained model"""
        features, labels = sample_generator.generate_random_data_sample(
            test_minibatch_size,
            self.input_dimension,
            self.output_dimension)

        self.trainer.test_minibatch({self.input: features, self.label: labels})

        output = softmax(z)
        result = output.eval({self.input: features})

        print("Label: \t\t", np.argmax(labels[:25], axis=1))
        print("Predicted: \t", np.argmax(result[0, :25, :], axis=1))

        # Visualizing the results
        print(self.model_parameters['b'].value)
        plotting.plot_results(self.model_parameters, labels, features)

    # a static method has the characteristic of NOT accesing any 'self' value of the class. So it can be called
    # without instantiating the class. Calling a static vs. calling a normal method:
    # Model.print_training_progress()
    # new_model = Model()
    # new_model.initialize_trainer()
    @staticmethod
    def print_training_progress(trainer: object, num_mini_batch: int, frequency: int, verbose=1)\
            -> [int, float, float]:
        """Prints the training progress."""
        # 'verbose = 1' in the function's argument is a default parameter, which the function applies when there is no
        # argument supplied that matches the parameter.
        training_loss, eval_error = "NA", "NA"

        if num_mini_batch % frequency == 0:
            training_loss = get_train_loss(trainer)
            eval_error = get_train_eval_criterion(trainer)

            if verbose:
                print("Minibatch: {0}, Loss {1:.4f}, Error: {2:.2f}".format(num_mini_batch, training_loss, eval_error))

        return num_mini_batch, training_loss, eval_error
