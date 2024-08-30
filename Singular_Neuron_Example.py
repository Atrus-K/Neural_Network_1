#a tenser is an object that can be represented in an array- here we will be using tensers in the form of arrays
#here we are coding a neuron, with the exmple of being fed the results of...
# 3 other neurons( outputs from neurons themselves from a previous layer ), or the "input layer"
inputs = [1,2,3,2.5]


weights = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]] #every neuron has weights 

biases= [2,3,0.5] #every neuron has a bias ( list of biases )

# first step for a neuron is to do inputs * weights + bias

layer_outputs = [] #output of current layer - 
for neuron_weights, neuron_bias in zip(weights,biases): # zip combines two lists into a list of lists element wise
    neuron_output = 0 # output of given neuron 
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output +-n_input*weight
    neuron_output =- neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)



