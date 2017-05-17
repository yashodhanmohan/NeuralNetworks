from numpy import mat, loadtxt, zeros, exp, multiply, sum, square, sqrt, hstack, unique, ones, argmax, power
from numpy.random import random
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

def activation(input_value, type):
    if type=='linear':
        return input_value
    if type=='sigmoid_bipolar':
        return 2.0 / (1 + exp(- input_value)) - 1
    if type=='sigmoid':
        return 1.0 / (1 + exp(- input_value))

def activation_derivative(output_value, type):
    if type=='linear':
        return 1
    if type=='sigmoid_bipolar':
        return 0.5 * multiply((1 + output_value), (1 - output_value))
    if type=='sigmoid':
        return multiply(output_value, (1 - output_value))

def error_derivative(target, output, loss_function):
    if loss_function=='least_squares':
        return target - output
    if loss_function=='fourth_power':
        return power(target - output, 3)
    if loss_function=='modified_least_squares':
        output[output>1] = 1
        output[output<-1] = -1
        return target - output
    if loss_function=='modified_fourth_power':
        output[output>1] = 1
        output[output<-1] = -1
        return power(target - output, 3)

def error(target, output, loss_function):
    if loss_function=='least_squares':
        return 0.5 * sum(square(target - output))
    if loss_function=='fourth_power':
        return 0.25 * sum(square(square(target - output)))
    if loss_function=='modified_least_squares':
        output[output>1] = 1
        output[output<-1] = -1
        return 0.5 * sum(square(target - output))
    if loss_function=='modified_fourth_power':
        output[output>1] = 1
        output[output<-1] = -1
        return 0.25 * sum(square(square(target - output)))

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
hid = inp+4
out = len(unique(list(data[:, -1])))
learning_rate = 1e-3
n_epoch = 900
input_layer_activation = 'linear'
hidden_layer_activation = 'sigmoid'
output_layer_activation = 'linear'
loss_function = 'least_squares'
# Initialize the algorithm parameters

x = data[:, 0:inp].T                                     # Input parameters
t = target_to_coded_class_labels(data[:, -1].T, out)     # Targets

# Initialize weights
W_xz = 0.001 * (mat(random(size=(inp, hid)))*2.0 - 1.0)
W_zy = 0.001 * (mat(random(size=(hid, out)))*2.0 - 1.0)

# W_xz = DW_xz =    inp x hid
# W_zy = DW_zy =    hid x out
# x    = X_out =    inp x n_train
# Z_in = Z_out =    hid x n_train
# Y_in = Y_out =    out x n_train

# Network : X -> Z -> Y

# Train the network
for epoch in range(n_epoch):

    X_in = x
    X_out = activation(X_in, input_layer_activation)
    
    Z_in = W_xz.T * x
    Z_out = activation(Z_in, hidden_layer_activation)
    
    Y_in = W_zy.T * Z_out
    Y_out = activation(Y_in, output_layer_activation)

    err_Y = error_derivative(t, Y_out, loss_function)
    err_H = W_zy * err_Y

    DW_zy = learning_rate * Z_out * multiply(err_Y, activation_derivative(Y_out, output_layer_activation)).T
    DW_xz = learning_rate * X_out * multiply(err_H, activation_derivative(Z_out, hidden_layer_activation)).T

    sumerr = error(t, Y_out, loss_function)
    W_xz = W_xz + DW_xz
    W_zy = W_zy + DW_zy

    print str(float(epoch)/n_epoch * 100) + ' % : ' + str(sqrt(sumerr/n_train)) + '\r',

print ''

# Validate the network
X_out = activation(x, input_layer_activation)
Z_out = activation(W_xz.T * X_out, hidden_layer_activation)
Y_out = activation(W_zy.T * Z_out, output_layer_activation)
error_training = error(t, Y_out, loss_function)
Y_out[Y_out>0] = 1
Y_out[Y_out<0] = -1
Y_out = coded_class_labels_to_target(Y_out)
t = coded_class_labels_to_target(t)
confusion = confusion_matrix(t, Y_out)
(overall_accuracy, average_accuracy, geometric_mean_accuracy) = accuracy(confusion)

print loss_function + ' error : ' + str(error_training/n_train)
print 'Confusion matrix : \n' + str(confusion)
print 'Overall accuracy : ' + str(overall_accuracy) + '%'
print 'Average accuracy : ' + str(average_accuracy) + '%'
print 'Geometric mean accuracy : ' + str(geometric_mean_accuracy) + '%'


# Test the network
test = mat(loadtxt(testing_file))
n_test = test.shape[0]
test_x = data[:, 0:inp].T
test_t = data[:, inp:inp+out].T
X_out = activation(test_x, input_layer_activation)
Z_out = activation(W_xz.T * X_out, hidden_layer_activation)
Y_out = activation(W_zy.T * Z_out, output_layer_activation)
