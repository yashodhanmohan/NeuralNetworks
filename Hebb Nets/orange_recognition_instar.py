import numpy as np

n_input = 4
n_output = 1
add_bias = True
n_epoch = 0
alpha = 0.4

def orange_test(w):
    p = np.mat([1, 0, 1, -1, -1]).T
    # Compute output response
    a = f((w.T * p)[0, 0])
    if a is 0:
        return False
    else:
        return True

# Activation function: hardlim
def f(y_in):
    if y_in > 0:
        return 1
    else:
        return 0

# Initialize weights
if add_bias:
    w = np.mat(np.zeros([n_input+1, n_output]))
else:
    w = np.mat(np.zeros([n_input, n_output]))

w[0, 0] = 1
w[1, 0] = 0
w[2, 0] = 0
w[3, 0] = 0
w[4, 0] = 0
p0 = 1
p1 = 1
p2 = -1
p3 = -1
t = 1

# Training - All inputs are oranges and hence target is always 1
while not orange_test(w):
    p0 = 1 - p0    
    n_epoch += 1
    # Prepare input vector
    p = np.mat([1, p0, p1, p2, p3]).T
    # Compute output response
    a = f((w.T * p)[0, 0])
    w = (1 - alpha * a) * w + alpha * a * p
    print w

print 'Weights are : ', w.T
print 'No. of epochs: ', n_epoch

# Testing
for p0 in [0, 1]:
    p = np.mat([1, p0, p1, p2, p3]).T
    a = f((w.T * p)[0, 0])
    print  p[1:, ].T, '=', a
