#example taken from https://www.geeksforgeeks.org/dilated-convolution/
import numpy as np
import tensorflow as tf
import sys
from  scipy.signal import convolve2d
 
np.random.seed(678)
tf.random.set_seed(6789)
sess = tf.compat.v1.Session()
 
# Initializing a 9x9 matrix of zeros.
mat_size = 9
matrix = np.zeros((mat_size,mat_size)).astype(np.float32)
 
# Assigning 1's in the middle of matrix
# to create a random input matrix
for x in range(4,7):
    for y in range(3,6):
        matrix[y,x] = 1
 
# Creating a random kernel for test
kernel = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]).astype(np.float32)
 
print("Original Matrix Shape : ",matrix.shape)
print(matrix)
print('\n')
print("Original kernel Shape : ",kernel.shape)
print(kernel)
 
# self-initializing a dilated kernel.
# ======[dilation factor = 3]======
dilated_kernel = np.array([
    [1,0,0,2,0,0,3],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [4,0,0,5,0,0,6],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [7,0,0,8,0,0,9]
])
 
print('\n')
print("Dilated kernel Shape : ",dilated_kernel.shape)
print(dilated_kernel)
 
print('\n')
print("DILATED CONVOLUTION RESULTS [Dilation Factor = 3]")
output = convolve2d(matrix,dilated_kernel,mode='valid')
print("Numpy Results Shape: ",output.shape)
print(output)