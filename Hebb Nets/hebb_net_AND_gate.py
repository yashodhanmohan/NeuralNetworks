import numpy as np

n_input = 2
n_output = 1
add_bias = True

# Initialize weights
if add_bias:
    w = np.mat(np.zeros([n_input+1, n_output]))
else:
    w = np.mat(np.zeros([n_input, n_output]))

# Training
for x1 in [-1, 1]:
    for x2 in [-1, 1]:
        # Prepare input vector
        x = np.mat([1, x1, x2]).T
        # Compute target vector
        t = 1 if (x1==1) and (x2==1) else -1
        w = w + (x *  t)

print 'Weights are : ', w.T

# Testing
for x1 in [1, -1]:
    for x2 in [1, -1]:
        # Prepare input vector
        x = np.mat([1, x1, x2]).T
        
        y = np.sign((w.T * x)[0, 0])
        print  x[1:, ].T, '=', y
        
