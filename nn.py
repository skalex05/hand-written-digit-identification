import numpy as np
import math
import pickle
import time


# Activation Functions

class Sigmoid:
    """
        Static Class.
        Used to determine how 'activated' a neuron becomes when input activations propagate through it.
        Includes the function itself f(x) as well as its derivative, f_prime(x)
    """

    @staticmethod
    def f(x):
        ex = np.exp(x)
        v = ex / (1 + ex)
        return np.nan_to_num(v)

    @staticmethod
    def f_prime(x):
        ex = np.exp(x)
        v = ex / (np.multiply(ex, (ex + 2)) + 1)
        return np.nan_to_num(v)


class ReLU:
    """
        Static Class.
        A simpler activation function that increases linearly with input when x>0 and is 0 when x <= 0.
        Includes the function itself f(x) as well as its derivative, f_prime(x)
    """

    @staticmethod
    def f(x):
        return np.maximum(0, x)

    @staticmethod
    def f_prime(x):
        return np.where(x > 0, 1, 0)


# Loss functions

class MeanSquaredError:
    """
        Static Class.
        Calculates a loss based upon how far off a set of activations are from the desired activations.
        Includes the function itself f(x) as well as its derivative, f_prime(x)
    """

    @staticmethod
    def f(activations, desired):
        if activations.shape != desired.shape:
            raise ValueError(f"Activation and desired matrices must be of the same size. "
                             f"{activations.shape} != {desired.shape}")
        return 0.5 * np.square(activations - desired)

    @staticmethod
    def f_prime(activations, desired):
        if activations.shape != desired.shape:
            raise ValueError(f"Activation and desired matrices must be of the same size"
                             f"{activations.shape} != {desired.shape}")
        return activations - desired


class Layer:
    """
        Consists of 'n' neurons.
        activation_function : defaults to Sigmoid
        random_bias : defaults to True (if false, biases will be set to 0)
    """

    def __init__(self, n_neurons, activation_function=Sigmoid, random_bias=True):
        """
            :param n_neurons: Determines how many neurons should be in this layer.
            :param activation_function: Determines how input activations propagate to this layer.
            :param random_bias: Determines if biases should be randomised by default.
        """
        self.n_neurons = n_neurons
        self.weights = None
        self.random_bias = random_bias
        if random_bias:
            self.biases = np.matrix((np.random.rand(n_neurons) - 0.5) * 2)
        else:
            self.biases = np.matrix(np.zeros(n_neurons))
        self.biases = self.biases.T
        self.activation_function = activation_function
        self.next = None
        self.prev = None
        self.z_j_sum = 0
        self.activation_sum = 0
        self.delta_preactivation_sum = None

    def reset(self):
        """
            Called by its associated NeuralNetwork Instance's function reset().
            This will reset all weights and biases.
        """

        if self.random_bias:
            self.biases = np.matrix((np.random.rand(self.n_neurons) - 0.5) * 2)
        else:
            self.biases = np.matrix(np.zeros(self.n_neurons))
        self.biases = self.biases.T

        # All non-output layers should reconnect to their next layer to reset weights.
        if self.next:
            self.connect_next(self.next)

    def connect_prev(self, prev_layer):
        """
            Set the previous layer attribute of this layer.
        """
        self.prev = prev_layer

    def connect_next(self, next_layer):
        """
            Set the next layer attribute of this layer.
            This will also randomise the weights associated between the two layers.
        """

        self.next = next_layer
        next_layer.weights = np.matrix((np.random.rand(next_layer.n_neurons, self.n_neurons) - 0.5) * 2)

    def propagate(self, activations):
        """
            Propagates the activations from the previous layer to the current layer.
            :return: activation vector for the current layer
        """
        if activations.shape != (self.prev.n_neurons, 1):
            raise ValueError(
                f"Layer expected activation matrix of shape {(self.prev.n_neurons, 1)} but got {activations.shape}")
        # Z_j variable stores the sum of weights multiplied by previous activations for each neuron in this layer.
        # Biases are also added.
        z_j = np.matmul(self.weights, activations) + self.biases
        self.z_j_sum += z_j
        self.activation_sum += activations

        return self.activation_function.f(z_j)

    def back_propagate(self, cost_function, learning_rate, batch_size, weight_penalty_lambda,
                       batch_activations=None, batch_desired=None):
        """
            Adjusts weights and biases of this layer.
             Used by the train function of a NeuralNetwork Instance to minimise the model's cost function.
        """

        # Calculate averages for z_j, the partial derivative of the cost function and mean activation.
        # This information will be used to find an average gradient to adjust the model towards.
        z_j = self.z_j_sum / batch_size
        f_prime_z_j = self.activation_function.f_prime(z_j)
        mean_activation = self.activation_sum / batch_size

        if type(batch_activations) is not type(None) and type(batch_desired) is not type(None):
            # This condition is for back propagating through the output layer of the network.
            # Here, the delta_preactivation_sum is calculated based upon an average of costs across batches.
            prime_costs = 0
            for i in range(batch_size):
                prime_costs += cost_function.f_prime(batch_activations[i], batch_desired[i].T)
            prime_costs /= batch_size
            self.delta_preactivation_sum = np.multiply(f_prime_z_j, prime_costs)
        else:
            # This branch will be used for all subsequent(hidden) layers of the network.
            if not self.next:
                raise AttributeError("Output layer back propagation must assign desired and output values")
            self.delta_preactivation_sum = np.multiply(f_prime_z_j,
                                                       np.matmul(self.next.weights.T,
                                                                 self.next.delta_preactivation_sum))

        self.z_j_sum = 0
        self.activation_sum = 0

        # Nudge weights and biases according to the calculated gradient of the cost function
        self.weights -= learning_rate * weight_penalty_lambda * self.weights / self.weights.shape[0]
        self.weights -= learning_rate * np.matmul(self.delta_preactivation_sum, mean_activation.T)
        self.biases -= learning_rate * self.delta_preactivation_sum


class NeuralNetwork:
    """
        This class is used to store all the layers which compose of a network.
        It includes functionality for training models and predicting output given an input vector.
    """

    def __init__(self, layers, cost_function, learning_rate, weight_penalty_lambda=0.0):
        """
            Ensure the number of neurons in the input and output layer correspond to the number of input/out datavalues
            :param layers: A list of layers to use (The first being input and last being output)
            :param cost_function: Defaults to MSE
            :param learning_rate: How fast a model fits to training data. Large values may cause instability.
            :param weight_penalty_lambda: How much strong weights should be penalised, reducing overfitting.
        """
        self.layers = layers
        self.input = self.layers[0]
        self.output = self.layers[-1]
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.weight_penalty_lambda = weight_penalty_lambda
        for i in range(len(layers)):
            if i > 0:
                layers[i].connect_prev(layers[i - 1])
            if i < len(layers) - 1:
                layers[i].connect_next(layers[i + 1])

    def propagate(self, input_activations):
        """
            Propagates input activations through the network and returns the output activations.
            :param input_activations:
            :return: Output Activations
        """
        activations = np.matrix(input_activations).T
        self.layers[0].activation_sum += activations
        for i in range(1, len(self.layers)):
            activations = self.layers[i].propagate(activations)
        return activations

    def reset(self):
        """
            Resets weights and biases for all layers
        """
        for layer in self.layers:
            layer.reset()

    def train(self, epochs, data, batch_size=1, k_folds=1, test_train_split=0.8, noise_p=0.0):
        """
            Trains a model on a set of data. Test/train split is automated within this function.
            For a model with n input neurons and m output neurons, Each row, representing a datapoint should consist of:
                 m label columns followed immediately by n observation columns.
            The function returns a set of lists of accuracies and loss over epochs, if you wanted to plot this data.

            :param batch_size: How many training samples are shown before back-propagation. Defaults to stochastic(1).
            :param epochs: How many times the model should view the training data.
            :param data: A numpy array. Read above regarding how data should be formatted.
            :param k_folds: k-fold cross validation assesses training performance with different train-test splits.
            :param test_train_split: How much data should go into the train dataset.
            :param noise_p: Data Augmentation: A parameter for adding noise to the train dataset to reduce overfitting.
            :return: avg_train_accuracy, test_accuracy, avg_loss, plt_epochs
        """
        st = time.time()
        # Shuffle the rows of data and split them into k, equally sized folds.
        np.random.shuffle(data)
        if k_folds > 1:
            folds = np.array_split(data, k_folds)
        else:
            folds = [data]

        # Create a set of arrays that will track training performance history
        avg_train_accuracy = np.zeros(epochs)
        avg_loss = np.zeros(epochs)
        plt_epochs = np.arange(epochs)
        test_accuracies = []

        for test_fold_i in range(k_folds):
            # K times, a model is trained using a different set of training folds and different test fold.
            self.reset()
            # Combine the train folds into a single array and select the test fold.
            if k_folds == 1:
                train_size = int(data.shape[0] * test_train_split)
                train_folds = folds[0][: train_size, :]
                test_fold = folds[0][train_size:, :]
            else:
                train_folds = folds[:test_fold_i] + folds[test_fold_i + 1:]
                train_folds = np.concatenate(train_folds)
                test_fold = folds[test_fold_i]

            # For a model with m output neurons and n input neurons.
            # Separate training labels (first m neurons) from data (remaining n neurons).
            training_data = np.matrix(train_folds[:, self.output.n_neurons:])
            training_data += np.random.normal(0, noise_p, training_data.shape)
            training_labels = np.matrix(train_folds[:, :self.output.n_neurons])
            n_examples = len(training_labels)

            # Separate training data and labels into batches
            n_batches = math.ceil(n_examples / batch_size)
            data_batches = np.split(training_data, n_batches)
            label_batches = np.split(training_labels, n_batches)

            for epoch in range(epochs):
                loss = 0
                correct = 0
                for i in range(n_batches):
                    # Track activations for each example in the batch
                    batch_activations = []
                    for k in range(batch_size):
                        # Propagate a training example through the network
                        output = self.propagate(data_batches[i][k])
                        batch_activations.append(output)
                        # Calculate the loss
                        loss += np.sum(self.cost_function.f(output, label_batches[i][k].T))
                        for layer in self.layers[-1:0:-1]:
                            loss += np.square(np.sum(layer.weights)) * self.weight_penalty_lambda * 0.5
                        # Tally how many examples were correctly identified
                        correct += np.all(abs(output - label_batches[i][k].T) < 0.5)

                    # Back propagate through the network, updating weights and biases to improve training performance.
                    self.layers[-1].back_propagate(self.cost_function, self.learning_rate, batch_size,
                                                   self.weight_penalty_lambda, batch_activations, label_batches[i])
                    for layer in self.layers[-2:0:-1]:
                        layer.back_propagate(self.cost_function, self.learning_rate, batch_size,
                                             self.weight_penalty_lambda)

                # Print current statistics for model performance to indicate how training has gone so far.
                loss /= n_examples
                accuracy = correct / n_examples
                avg_train_accuracy[epoch] += accuracy
                avg_loss[epoch] += loss
                if k_folds > 1:
                    print(f"Epoch {epoch + 1} of Fold {test_fold_i + 1}/{k_folds}: Average Train Loss:"
                          f" {loss} Average Train Accuracy {accuracy}")
                else:
                    print(f"Epoch {epoch + 1}: Average Train Loss: {loss} Average Train Accuracy {accuracy}")

            # Once trained, test the model on unseen data and output performance

            testing_data = test_fold[:, self.output.n_neurons:]
            testing_labels = test_fold[:, :self.output.n_neurons]

            t = 0
            for i in range(len(testing_data)):
                answer = self.propagate(testing_data[i])[0, 0]
                t += np.all(abs(answer - testing_labels[i]) < 0.5)
            test_accuracy = t / len(test_fold)
            test_accuracies.append(test_accuracy)

            print("~~~ Test ~~~")
            if k_folds > 1:
                print(f"Fold {test_fold_i + 1} Test Accuracy: {test_accuracy}")
            else:
                print(f"Test Accuracy: {test_accuracy}")
            pickle.dump(self, open(f"Fold{test_fold_i}.nn", "wb"))

        # Output final training results

        test_high = max(test_accuracies)
        test_low = min(test_accuracies)
        test_accuracy = sum(test_accuracies) / k_folds
        avg_train_accuracy /= k_folds
        avg_loss /= k_folds

        if k_folds > 1:
            print(f"Average Test Accuracy Across {test_accuracy} (High: {test_high} Low: {test_low})")
        else:
            print(f"Average Test Accuracy Across {test_accuracy}")
        print(f"Training time: {round(time.time() - st, 2)} seconds")

        return avg_train_accuracy, test_accuracy, avg_loss, plt_epochs

    def predict(self, data):
        """
            This function returns the model's outputs for an array of input data.
            :param data: A np array of shape (j x n) for a network with n input neurons. j is the number of rows.
            :return: A np array of shape (j x m) for a network with m output neurons. j is the number of rows.
        """
        output = np.matrix(np.zeros((data.shape[0], self.output.n_neurons)))
        for i in range(data.shape[0]):
            output[i] = self.propagate(data[i]).T
        return output
