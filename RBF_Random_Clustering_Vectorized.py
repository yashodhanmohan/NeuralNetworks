from numpy import mat, loadtxt, zeros, exp, multiply, sum, square, sqrt, hstack, unique, ones, argmax, power, argmin
from numpy.random import random, choice, randint
from numpy.linalg import pinv
from sys import argv
import matplotlib.pyplot as plt

def target_to_coded_class_labels(t, n_classes):
    m = -1 * mat(ones((n_classes, t.shape[1])))
    for i in range(t.shape[1]):
        m[int(t[0, i]-1), i] = 1
    return m

def coded_class_labels_to_target(m):

    return argmax(m, axis=0) + 1

def confusion_matrix(target, output):
    confusion = mat(zeros([len(unique(list(target))), len(unique(list(target)))]))
    for i in range(target.shape[1]):
        confusion[int(target[0, i])-1, int(output[0, i])-1] += 1
    return confusion

def accuracy(confusion_matrix):
    overall_accuracy = (confusion_matrix.trace() / confusion_matrix.sum()) * 100
    average_accuracy = 0
    geometric_mean_accuracy = 1
    for i in range(confusion_matrix.shape[0]):
        average_accuracy += (confusion_matrix[i, i] / confusion_matrix.sum(axis=1)[i, 0])
        geometric_mean_accuracy *= (100.0 * (confusion_matrix[i, i] / confusion_matrix.sum(axis=1)[i, 0]))
    average_accuracy *= (100.0 / confusion_matrix.shape[0])
    geometric_mean_accuracy = geometric_mean_accuracy**(1.0/confusion_matrix.shape[0])
    return (overall_accuracy[0, 0], average_accuracy, geometric_mean_accuracy)

training_file = argv[1]
testing_file = argv[2]

data = mat(loadtxt(training_file))
n_train = data.shape[0]

# Initialize the algorithm parameters
inp = data.shape[1] - 1
hid = inp + 2
out = len(unique(list(data[:, -1])))
cluster_centers = True
mu = 1e-2
# Initialize the algorithm parameters

x = data[:, 0:inp].T                                     # Input parameters
t = target_to_coded_class_labels(data[:, -1].T, out)     # Targets

# Initialize weights
center = x[:, choice(x.shape[1], hid, replace=False)].copy()
if cluster_centers:
    one = ones((1, hid))
    count = 2000
    while count > 0:
        X = x[:, randint(n_train)]
        center_to_update = argmin(power(sum(square(center - (X * one)), axis=0), 0.5))
        center[:, center_to_update] = (1 - mu) * center[:, center_to_update] + mu * X
        count -= 1

d_max = 0.0
for i in range(hid):
    for j in range(i):
        d = power(sum(power(center[:, i] - center[:, j], 2)), 0.5)
        if d_max < d:
            d_max = d
spread = (d_max / sqrt(hid)) * mat(ones((1, hid)))
W_zy = 0.001 * (mat(random(size=(hid, out)))*2.0 - 1.0)

# W_zy = DW_zy =    hid x out
# x    = X_out =    inp x n_train
# Z_in = Z_out =    hid x n_train
# Y_in = Y_out =    out x n_train

# Network : X -> Z -> Y
# Train the network

X_out = x

Z_in = mat(zeros((hid, n_train)))
one = ones((1, hid))
for i in range(n_train):
    Z_in[:, i] = (power(sum(square(center - (X_out[:, i] * one)), axis=0), 0.5) / (2 * power(spread, 2))).T
Z_out = exp(-Z_in)

W_zy = (t * pinv(Z_out)).T
Y_out = W_zy.T * Z_out

Y_out[Y_out<0] = -1
Y_out[Y_out>0] = 1
t = coded_class_labels_to_target(t)
Y_out = coded_class_labels_to_target(Y_out)
confusion = confusion_matrix(t, Y_out)
(overall_accuracy, average_accuracy, geometric_mean_accuracy) = accuracy(confusion)

print confusion
print 'Overall accuracy : ' + str(overall_accuracy)
print 'Average accuracy : ' + str(average_accuracy)
print 'Geometric mean accuracy : ' + str(geometric_mean_accuracy)