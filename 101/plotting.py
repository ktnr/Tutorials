import matplotlib.pyplot as plt


def plot_input_data(labels: [int], features: [int]):
    """Plots the input data."""
    # Given this is a 2 class
    colors = ['r' if l == 0 else 'b' for l in labels[:, 1]]

    plt.scatter(features[:, 0], features[:, 1], c=colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")
    plt.show()


def plot_training_progress(plot_data: '{string: float}'):
    """Plots the training progress."""
    plt.figure(1)

    plt.subplot(211)
    plt.plot(plot_data["batchsize"], plot_data["avgloss"], "b--")
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training Loss")

    plt.subplot(212)
    plt.plot(plot_data["batchsize"], plot_data["avgerror"], "r--")
    plt.xlabel("Minibatch number")
    plt.ylabel("Label Prediction Error")
    plt.title("Minibatch run vs. Label Prediction Error")

    plt.show()


def plot_results(model_parameters: '{string : parameter}', labels: [int], features: [int]):
    """Plots the training results."""
    bias_vector = model_parameters['b'].value
    weight_matrix = model_parameters['w'].value

    # Plot the data
    colors = ['r' if l == 0 else 'b' for l in labels[:, 0]]
    plt.scatter(features[:, 0], features[:, 1], c=colors)
    plt.plot([0, bias_vector[0] / weight_matrix[0][1]],
             [bias_vector[1] / weight_matrix[0][0], 0], c='g', lw=3)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")
    plt.show()
