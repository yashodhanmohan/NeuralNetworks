import numpy as np

n_input = 2
n_output = 1
add_bias = True
n_epoch = 0

def banana_test(w):
    p1 = 1
    for p0 in [0, 1]:
        # Prepare input vector
        p = np.mat([1, p0, p1]).T
        # Compute output response
        a = f((w.T * p)[0, 0])
        if a==0:
            return False
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

w[0, 0] = -0.5
w[1, 0] = 1
w[2, 0] = 0
p1 = 1
t = 1

# Training - All inputs are bananas and hence target is always 1
while not banana_test(w):
    for p0 in [0, 1]:
        if not banana_test(w):
            n_epoch += 1
            # Prepare input vector
            p = np.mat([1, p0, p1]).T
            # Compute output response
            a = f((w.T * p)[0, 0])
            w = w + a * p

print 'Weights are : ', w.T
print 'No. of epochs: ', n_epoch

# Testing
for p0 in [0, 1]:
    p = np.mat([1, p0, p1]).T
    a = f((w.T * p)[0, 0])
    print  p[1:, ].T, '=', a
