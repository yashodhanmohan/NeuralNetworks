from numpy import mat, ones, count_nonzero
from numpy.random import random

def f(A):
    A[A<=0] = 0
    return A

inp = 6
epsilon = 1.0/(2 * inp)

# Set random activations
A = mat(random(inp)).T

# Set weights for the net
W = mat(-epsilon * ones((inp, inp)))
for i in range(inp):
    W[i, i] = 1

while count_nonzero(A)>1:
    A = f(W*A)