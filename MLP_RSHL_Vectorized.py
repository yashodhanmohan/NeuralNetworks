from numpy import delete, insert, array, asarray, mat, loadtxt, zeros, exp, multiply, sum, square, sqrt, hstack, vstack, unique, ones, argmax, power, logical_and, bincount
from numpy.random import random, randint
from numpy.linalg import norm
from sys import argv
from math import floor
import matplotlib.pyplot as plt

def target_to_coded_class_labels(t, n_classes):
    m = -1 * mat(ones((n_classes, t.shape[1])))
    for i in range(t.shape[1]):
        m[int(t[0, i]), i] = 1
    return m

def coded_class_labels_to_target(m):
    return argmax(m, axis=0) + 1

def confusion_matrix(target, output):
    confusion = mat(zeros([len(unique(list(target))), len(unique(list(target)))]))
    for i in range(target.shape[1]):
        confusion[int(target[0, i])-1, int(output[0, i])-1] += 1
    return confusion

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

def stratified_folds(data, k):
    data = data.copy()
    data[:, -1] = data[:, -1] - 1
    folds = []
    classwise_data = []
    classwise_data_distribution = []
    n_classes = len(unique(array(data[:, -1]).flatten()))
    n_data = data.shape[0]
    size_1_fold = floor(float(n_data) / k)
    for i in range(n_classes):
        classdata = data[array(data[:, -1]==i).flatten(), :].copy()
        classdata = insert(classdata, classdata.shape[1], 0, axis=1)
        classdata[:, -1] = mat(randint(low=0, high=k, size=classdata.shape[0])).T
        classwise_data.append(classdata)
        classwise_data_distribution.append(classwise_data[i].shape[0])
    for i in range(k):
        for j in range(n_classes):
            if j==0:
                fold = classwise_data[j][array(classwise_data[j][:, -1]==i).flatten(), :]
            else:
                fold = vstack([fold, classwise_data[j][array(classwise_data[j][:, -1]==i).flatten(), :]])
        fold = delete(fold, -1, 1)
        folds.append(fold.copy())
    return folds

training_file = argv[1]
testing_file = argv[2]
testing_result_file = argv[3]

data_train = mat(loadtxt(training_file))
data_test = hstack([mat(loadtxt(testing_file)), mat(loadtxt(testing_result_file)).T])
n_train = data_train.shape[0]
n_test = data_test.shape[0]

# Initialize the algorithm parameters
inp = data_train.shape[1] - 1
hid = inp + 6
out = len(unique(list(data_train[:, -1])))
learning_rate = 1e-4
n_epoch = 400
n_fold = 10
print 'Input = ' + str(inp)
print 'Hidden = ' + str(hid)
print 'Output = ' + str(out)
# Initialize the algorithm parameters

# Initialize weights
data_fold = stratified_folds(data_train, n_fold)
W_xz = (mat(random(size=(inp, hid)))*2.0 - 1.0)
W_zy = (mat(random(size=(hid, out)))*2.0 - 1.0)
beta = mat(ones(out)).T / out
# m = mat(ones((out, 1))) * beta.T
# VC
m = mat([[ 0.,    0.,    0.,    0.],
        [  30.,   0.,   26.,    7.],
        [  24.,   21.,   0.,    3.],
        [   5.,    0.,    0.,   0.]])
# PIMA
# m = mat([[ 0.,   56.],
#         [  46.,   0.]])
# Liver
# m = mat([[ 0.,  16.],
#         [ 32.,  0.]])
# ION
# m = mat([[ 0.,    7.],
#         [  38.,   0.]])
# Iris
# m = mat([[ 0.,   0.,   0.],
#         [  8.,   0.,  24.],
#         [  0.,   0.,  0.]])
# ae
# m = mat([[ 0.,   0.,   0.,   0.],
#         [  0.,  0.,   0.,   0.],
#         [  2.,   0.,  0.,   0.],
#         [  0.,   0.,   0.,  0.]])
# Wine
# m = mat([[ 0.,   0.,   0.],
#         [  4.,  0.,   0.],
#         [  1.,   1.,  0.]])

m = m / norm(m)

ones_out = mat(random(out)).T
epsilon = 1e-3

# Network : X -> Z -> Y

# Train the network
for k in range(n_fold):

    # Prepare training data and CV data
    data = vstack(data_fold[:k] + data_fold[k+1:])
    cv = data_fold[k]
    n_train = data.shape[0]
    # Extract input matrix from training data
    x = data[:, 0:inp].T
    # Extract target outputs from training data
    target = data[:, -1].T 
    t = target_to_coded_class_labels(target, out)     # Targets

    M = mat(zeros((out, n_train)))
    for i in range(n_train):
        M[:, i] = m[:, int(target[0, i])]

    N = mat(bincount(list(asarray(target[0, :])[0]))).T

    print 'CV for : ' + str(k)
    err = 0
    for epoch in range(n_epoch):

        X_in = x
        X_out = X_in
        
        Z_in = W_xz.T * X_out
        Z_out = 1.0 / (1 + exp(- Z_in))
        
        Y_in = W_zy.T * Z_out
        Y_out = Y_in

        t_Y_out = multiply(t, Y_out)

        X_case1 = - power(M - 1, 2)
        X_case2 = 2 * multiply((multiply(multiply(M, t), Y_out) - 1), M)
        X_case3 = 2 * (multiply(t, Y_out) - 1)
        X_case4 = 0 * t

        X_case1[t_Y_out >= -1] = 0
        X_case2[logical_and(t_Y_out <= -1, t_Y_out > 0)] = 0
        X_case3[logical_and(t_Y_out <=  0, t_Y_out > 1)] = 0
        X_case4[t_Y_out <= 1] = 0

        X = multiply((X_case1 + X_case2 + X_case3 + X_case4), t)

        # Weight updation
        DW_xz = X_out * multiply(multiply(Z_out, 1 - Z_out), (W_zy * X)).T
        DW_zy = Z_out * X.T

        W_xz = W_xz - learning_rate * DW_xz
        W_zy = W_zy - learning_rate * DW_zy

        err_this_epoch = sqrt(sum(square(t - Y_out))/n_train)
        err = err + err_this_epoch

        # Conditional probability calculation
        P = (Y_out + 1.0) / 2.0
        P[P<=epsilon] = epsilon
        P[P>1] = 1

        m = multiply((ones_out * (beta / N).T), (ones_out * N.T + (1.0/P) * P.T))
        m = m / norm(m)
        for i in range(n_train):
            M[:, i] = m.T[:, int(target[0, i])]

        print str(float(epoch)/n_epoch * 100) + ' % : ' + str(sqrt(err_this_epoch/n_train)) + '\r',
    print ''

    X_in = cv[:, 0:inp].T
    X_out = X_in
    
    Z_in = W_xz.T * X_out
    Z_out = 1.0 / (1 + exp(- Z_in))
    
    Y_in = W_zy.T * Z_out
    Y_out = Y_in

    t = target_to_coded_class_labels(cv[:, -1].T, out)

    print 'Error on testing fold : ' + str(sqrt(sum(square(t - Y_out))/n_train))

# Validation on training phase
x = data_train[:, 0:inp].T
target = data_train[:, -1].T - 1 
t = target_to_coded_class_labels(target, out)

X_in = x
X_out = x

Z_in = W_xz.T * x
Z_out = 1.0 / (1 + exp(- Z_in))

Y_in = W_zy.T * Z_out
Y_out = Y_in

Y_out[Y_out>0] = 1
Y_out[Y_out<0] = -1

Y_out = coded_class_labels_to_target(Y_out)
t = coded_class_labels_to_target(t)

for j in list(asarray(Y_out)[0]):
    print j
 

confusion = confusion_matrix(t, Y_out)
(overall_accuracy, average_accuracy, geometric_mean_accuracy) = accuracy(confusion)

print '\n\n========== Training phase results ====================\n'
print 'No. of samples : ' + str(data_test.shape[0])
print 'Confusion matrix : \n' + str(repr(confusion))
print 'Overall accuracy : ' + str(overall_accuracy) + '%'
print 'Average accuracy : ' + str(average_accuracy) + '%'
print 'Geometric mean accuracy : ' + str(geometric_mean_accuracy) + '%'

# Testing phase
x = data_test[:, 0:inp].T
target = data_test[:, -1].T - 1 
t = target_to_coded_class_labels(target, out)

X_in = x
X_out = x

Z_in = W_xz.T * x
Z_out = 1.0 / (1 + exp(- Z_in))

Y_in = W_zy.T * Z_out
Y_out = Y_in

Y_out[Y_out>0] = 1
Y_out[Y_out<0] = -1

Y_out = coded_class_labels_to_target(Y_out)
t = coded_class_labels_to_target(t)

for j in list(asarray(Y_out)[0]):
    print j

confusion = confusion_matrix(t, Y_out)
(overall_accuracy, average_accuracy, geometric_mean_accuracy) = accuracy(confusion)

print '\n\n========== Testing phase results ====================\n'
print 'No. of samples : ' + str(data_test.shape[0])
print 'Confusion matrix : \n' + str(repr(confusion))
print 'Overall accuracy : ' + str(overall_accuracy) + '%'
print 'Average accuracy : ' + str(average_accuracy) + '%'
print 'Geometric mean accuracy : ' + str(geometric_mean_accuracy) + '%'