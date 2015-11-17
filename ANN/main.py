from __future__ import print_function
import numpy as np

from NeuralNet import NeuralNet

file_lines = []

# read in the file and save it line by line as a list of strings
with open ('optdigits_train.txt', 'r') as training_file:
    file_lines = training_file.readlines()

inputs = [[float(x) for x in line.strip().split(',')[0:-1]] for line in file_lines]
answers = [int(x) for x in line.strip().split(',')[-1] for line in file_lines]

[array.append(1.0) for array in inputs]  # add the bias nodes to the input



net = NeuralNet()
net.train(inputs[0:10], answers[0:10])
