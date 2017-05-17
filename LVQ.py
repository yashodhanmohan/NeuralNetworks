from numpy import mat, loadtxt, vstack, ones, argmin, count_nonzero, square, sum, zeros, unique
from numpy.random import random
from sys import argv

# Program for LVQ Classifier
training_file = argv[1]
test_file = argv[2]

# Load data
data = mat(loadtxt(training_file)).T

inp = data.shape[0] - 1 # Number of inputs - feature vectors which decide clusters
out = len(unique(list(data[-1, :]))) # Number of outputs - clusters into which data is to be divided
alpha = 1e-3
n_epoch = 1000

# Set initial random weights assuming random centers for clusters
W = mat(random(size=(inp, out)))

# Extract input and target output from data
n_train = data.shape[1]
x = data[0:inp, :]
t = data[inp, :]

# Initialize confusion matrix
C = mat(zeros((out, out)))

# W : inp x out
# x : inp x ndata
# t : 1   x ndata
# A : out x ndata

# Update centers
for n in range(n_epoch):
    print str(float(n)/n_epoch * 100) + '%\r',
    for i in range(n_train):
        d = sum(square(W - (x[:, i] * mat(ones(out)))), axis=0)
        J = argmin(d)
        T = t[0, i] - 1
        if T == J:
            W[:, J] = (1 - alpha) * W[:, J] + alpha * x[:, i]
        else:
            W[:, J] = (1 + alpha) * W[:, J] - alpha * x[:, i]

actual_N = mat(zeros(out)).T
for i in range(n_train):
    T = int(t[0, i] - 1)
    J = int(argmin(sum(square(W - (x[:, i] * mat(ones(out)))), axis=0)))
    actual_N[T, 0] = actual_N[T, 0] + 1
    C[T, J] = C[T, J] + 1

average_accuracy = 0
geometric_mean_accuracy = 1
for i in range(out):
    average_accuracy += C[i, i]/actual_N[i, 0]
    geometric_mean_accuracy *= (100 * C[i, i]/actual_N[i, 0])
average_accuracy *= (100/out)
geometric_mean_accuracy = geometric_mean_accuracy**(1.0/out)

print W
print C
print 'Overall accuracy : ' + str(100*C.trace()/n_train)
print 'Average accuracy : ' + str(average_accuracy)
print 'Geometric mean accuracy : ' + str(geometric_mean_accuracy)