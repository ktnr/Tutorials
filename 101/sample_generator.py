import numpy as np

# Set the seed for the random number generator
np.random.seed(0)


def generate_random_data_sample(sample_size: int, feature_dimension: int, num_classes: int)\
        -> [float, [float]]:
    """Generates a random data sample."""
    # Create synthetic data using numpy
    y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    x = (np.random.randn(sample_size, feature_dimension) + 3) * (y + 1)

    # Specify the data type to match the input variable used later in the tutorial
    # Default type from numpy.random.randn is double
    x = x.astype(np.float32)

    # Converting class 0 into the vector    "1  0   0"
    # Class 1 into vector "                 "0  1   0", ...
    class_indexes = [y == class_number for class_number in range(num_classes)]
    y = np.asarray(np.hstack(class_indexes), dtype=np.float32)

    return x, y
