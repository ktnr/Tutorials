# Project
import plotting
import sample_generator
import model

# Set the size of the network to be trained
input_dimension = 2
output_dimension = 2
sample_size = 32
features, labels = sample_generator.generate_random_data_sample(sample_size, input_dimension, output_dimension)

plotting.plot_input_data(labels, features)


# Define a dictionary to store the model parameters
# info: A dictionary is a (key, value)-collection. One can supply the key and get the value in return like so:
# test = modelParameters["w"]
# test == None
# [out]: true
model_parameters = {"w": None, "b": None}

# Create the model and its components
model = model.Model(model_parameters, input_dimension, output_dimension)
z = model.linear_layer()
model.initialize_trainer(z, output_dimension)

# Run the trainer and perform the training
minibatch_size = 25
num_samples_to_train = 20000
plot_data = model.train(minibatch_size, num_samples_to_train)

plotting.plot_training_progress(plot_data)

# Run the trained model on a newly generated data set in order to validate the results produced by our model
test_minibatch_size = 25
model.test_trained_model(z, test_minibatch_size)
