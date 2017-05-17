from numpy import mat, loadtxt, zeros, exp, multiply, sum, square, sqrt, hstack, linspace, vstack
from numpy.random import random

# Program for MLP
# Update weights for an epoch

data = mat(loadtxt('her.tra'))
n_train = data.shape[0]

# Initialize the algorithm parameters
inp = 2
hid = 6
out = 1
lam = 1e-2
n_epoch = 90000

x = data[:, 0:inp].T                # Input parameters
t = data[:, inp:inp+out].T          # Targets

# Initialize weights
W_xz = 0.001 * (mat(random(size=(inp, hid)))*2.0 - 1.0)
W_zy = 0.001 * (mat(random(size=(hid, out)))*2.0 - 1.0)

# W_xz = DW_xz =    inp x hid
# W_zy = DW_zy =    hid x out
# x =               inp x n_train
# Z_in = Z_out =    hid x n_train
# Y_in = Y_out =    out x n_train

# Network : X -> Z -> Y

# Train the network
for epoch in range(n_epoch):
    # Prepare inputs for each neuron in hidden layer (Z)
    Z_in = W_xz.T * x
    # Pass through activation function (binary sigmoid) and prepare output of hidden layer
    Z_out = 1.0 / (1 + exp(-Z_in))

    # Prepare input for each neuron in output layer (Y)
    Y_in = W_zy.T * Z_out
    # Pass through activation function (identity) and prepare output of output layer
    Y_out = Y_in

    # Calculate delta for ZY weight matrix
    DW_zy = lam * Z_out * (t - Y_out).T

    # Calculate delta for XZ weight matrix
    DW_xz = lam * x * multiply(multiply(W_zy * (t - Y_out), Z_out), 1-Z_out).T
    
    # Sum of squared errors in output
    sumerr = sum(square(t - Y_out))

    # Update weight matrices XZ and ZY
    W_xz = W_xz + DW_xz
    W_zy = W_zy + DW_zy

    print sqrt(sumerr/n_train)

# Validate the network
rms_training = mat(zeros((out, 1)))
res_training = mat(zeros((n_train, 2)))
Z_out = 1.0 / ( 1 + exp(-W_xz.T * x))
Y_out = W_zy.T * Z_out
rms_training = rms_training + sum(square(t-Y_out), axis=1)
res_training = hstack([t.T, Y_out.T])
print sqrt(rms_training/n_train)

# Test the network
test = mat(loadtxt('her.tes'))
n_test = test.shape[0]
test_x = data[:, 0:inp].T
test_t = data[:, inp:inp+out].T
Z_out = 1.0 / ( 1 + exp(-W_xz.T * test_x))
Y_out = W_zy.T * Z_out
rms_testing = sum(square(test_t-Y_out), axis=1)
res_testing = hstack([test_t.T, Y_out.T])
print sqrt(rms_training/n_test)