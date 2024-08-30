#reasons for batches- allows for parallel operations, 
#neural network processing is usually done on gpus
#batches allows for generalisation - allows us to send matches of samples,
#not just one sample at a time, we can send multiple samples so that it can process more datapoints.
#e.g. a line of best fit is easier to do when you have more samples vs just one.

#"why not all the samples at a time?", the line of best fit will be hurt, as whilst in sample data line of best fit...
#will be good, any out of sample data will be skewed.
# best batch sizes are 32, 64 or 128

# remember for matrix for dot products x,y must be with a y,z matrix size,
# if needed, we can solve a shape error by transposing a matrix

#in this case, the inputs and the weights matrixes are the same, as a result of this we need to transpose the weights matrix such that it fits...
#the dimentions of the inputs matrix, because we have done this, we need to swap the order of the dot product multiplication to match this,
# the reason we do this and not just transpose the inputs is so that 1) it would save having to transpose all of the inputs, 2) it is standard neural network convention, 
#3) it results in a matrix that is the appropriate size., e.g. in this instane a 3x4 matrix as opposed to a 3x3.



import numpy as np

inputs = [[1,2,3,2.5], 
          [2.0, 5.0, -1.0, 2.0],
           [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

biases1 = [2, 3, 0.5]

weights2 = [[0.1,-0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer_1_outputs = np.dot(inputs, np.array(weights1).T) + biases1 #outputs for layer 1 are going to be the inputs for layer 2

layer_2_outputs = np.dot(layer_1_outputs, np.array(weights2).T) + biases2

print( layer_2_outputs)