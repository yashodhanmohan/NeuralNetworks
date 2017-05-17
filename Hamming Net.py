from numpy import mat, loadtxt, vstack, ones, argmax, count_nonzero
from numpy.random import random

# Program for MLP Classifier

inp = 4 # Number of inputs - feature vectors which decide clusters
out = 3 # Number of outputs - clusters into which data is to be divided
alpha = 1e-3
n_epoch = 9000

# Set initial random weights assuming random centers for clusters
W = mat(random(size=(inp+1, out)))

# Load data
data = mat(loadtxt('iris.tra')).T
# Add all-1 input to data
data = vstack([mat(ones(data.shape[1])), data])

# Extract input and target output from data
x = data[0:inp+1, :]
t = data[inp+1, :]
n_train = data.shape[1]


# W : inp+1 x out
# x : inp+1 x ndata
# t : 1     x ndata
# A : out   x ndata


# Update centers
for n in range(n_epoch):
    # Get activation for all inputs
    A = W.T * x
    # Pass through competitive network - Winner take all
    max_indices = argmax(A, axis=0)
    for i in range(n_train):
        A[:, i] = 0
        A[max_indices[0, i], i] = 1
    # W = W + alpha * ((x - W*A) * A.T)
    for i in range(n_train):
        W[:, max_indices[0, i]] = (1 - alpha) * W[:, max_indices[0, i]] + alpha * x[:, i]


# Validation
A = W.T * x
max_indices = argmax(A, axis=0)
for i in range(n_train):
    A[:, i] = 0
    A[max_indices[0, i], i] = 1
print W
print count_nonzero(max_indices+1==t)