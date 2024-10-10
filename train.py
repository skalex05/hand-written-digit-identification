import numpy as np
from mnist import MNIST
from nn import NeuralNetwork, Layer, ReLU, MeanSquaredError

# Retrieve MNIST dataset for handwritten digits
print("Retrieving data")
data = MNIST(return_type="numpy")

train_img, train_labels = data.load_training()
test_img, test_labels = data.load_testing()

print(train_img.shape, test_img.shape)
print(train_labels.shape, test_labels.shape)

print("Combining test and train")
images = np.concatenate((train_img, test_img))
labels_pre_proc = np.concatenate((train_labels, test_labels))

labels = np.zeros((labels_pre_proc.shape[0], 10))
print("Reformatting labels")
for i in range(labels_pre_proc.shape[0]):
    label = labels_pre_proc[i]
    labels[i, label] = 1

data = np.concatenate((labels, images), axis=1)
data = data[:5000, :]

print("Building Network")
model = NeuralNetwork([
    Layer(784),
    Layer(128, ReLU),
    Layer(64, ReLU),
    Layer(32, ReLU),
    Layer(10)
], MeanSquaredError, 0.001)

model.train(10, data)
