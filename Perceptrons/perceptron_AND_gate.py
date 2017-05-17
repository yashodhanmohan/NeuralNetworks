import numpy as np

n_input = 2
n_output = 1
add_bias = True
n_epoch = 0
theta = 0.2         # Threshold
alpha = 1           # Learning rate

# Initialize weights
if add_bias:
    w = np.mat(np.zeros([n_input+1, n_output]))
else:
    w = np.mat(np.zeros([n_input, n_output]))

# Define activation function
def f(y_in, theta):
    if y_in > theta:
        return 1
    elif y_in < -theta:
        return -1
    else:
        return 0

def and_gate_test(w):
    for x1 in [1, 0]:
        for x2 in [1, 0]:
            x = np.mat([1, x1, x2]).T
            y_in = (w.T * x)[0, 0]
            y = f(y_in, theta)
            t = 1 if x1 and x2 else -1
            if y!=t:
                return False
    return True

# Training
while not and_gate_test(w):
    n_epoch += 1
    for x1 in [1, 0]:
        for x2 in [1, 0]:
            # Prepare input vector
            x = np.mat([1, x1, x2]).T
            # Compute target vector
            t = 1 if x1 and x2 else -1
            # Compute net input
            y_in = (w.T * x)[0, 0]
            # Compute response
            y = f(y_in, theta)
            # Update weights if response and target are not same
            if y != t:
                w = w + alpha * x *  t
            # print  x[1:, ].T, '=', y, '~', t, ':', w.T

print 'Weights are : ', w.T
print 'Number of epochs: ', n_epoch
for x1 in [1, 0]:
    for x2 in [1, 0]:
        x = np.mat([1, x1, x2]).T
        y_in = (w.T * x)[0, 0]
        y = f(y_in, theta)
        print  x[1:, ].T, '=', y