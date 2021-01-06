import random
import math

def sigmoid(z):
    return 1/(1+math.pow(math.e, z))

input = [1,1]
target = [0]

w1 = random.uniform(0, 1)
w2 = random.uniform(0, 1)
w3 = random.uniform(0, 1)
w4 = random.uniform(0, 1)
w5 = random.uniform(0, 1)
w6 = random.uniform(0, 1)
w7 = random.uniform(0, 1)
w8 = random.uniform(0, 1)
w9 = random.uniform(0, 1)
neuron1 = input[0] * w1 + input[1] * w4
neuron2 = input[0] * w2 + input[1] * w5
neuron3 = input[0] * w3 + input[1] * w6
#forward propogate
output = sigmoid(neuron1) * w7 + sigmoid(neuron2) * w8 + sigmoid(neuron3) * w9
#backword propogate
deltaOutputSum = output * (target[0]-output)
neuron1WSum = deltaOutputSum / sigmoid(neuron1)
neuron2WSum = deltaOutputSum / sigmoid(neuron2)
neuron3WSum = deltaOutputSum / sigmoid(neuron3)
deltaHiddenSum1Neuron = (deltaOutputSum / (deltaOutputSum / w7)) * sigmoid(neuron1)
deltaHiddenSum2Neuron = (deltaOutputSum / (deltaOutputSum / w8)) * sigmoid(neuron2)
deltaHiddenSum3Neuron = (deltaOutputSum / (deltaOutputSum / w9)) * sigmoid(neuron3)
neuron1 += deltaHiddenSum1Neuron
neuron2 += deltaHiddenSum2Neuron
neuron3 += deltaHiddenSum3Neuron
dW1 = deltaHiddenSum1Neuron / (input[0] * input[1])
dW2 = deltaHiddenSum2Neuron / (input[0] * input[1])
dW3 = deltaHiddenSum3Neuron / (input[0] * input[1])
w1 += dW1
w2 += dW2
w3 += dW3
w4 += dW1
w5 += dW2
w6 += dW3



for i in range(100000):
    # forward propogate
    output = sigmoid(neuron1) * w7 + sigmoid(neuron2) * w8 + sigmoid(neuron3) * w9
    # backword propogate
    deltaOutputSum = output * (target[0] - output)
    neuron2WSum = deltaOutputSum / sigmoid(neuron2)
    neuron3WSum = deltaOutputSum / sigmoid(neuron3)
    deltaHiddenSum1Neuron = (deltaOutputSum / (deltaOutputSum / w7)) * sigmoid(neuron1)
    deltaHiddenSum2Neuron = (deltaOutputSum / (deltaOutputSum / w8)) * sigmoid(neuron2)
    deltaHiddenSum3Neuron = (deltaOutputSum / (deltaOutputSum / w9)) * sigmoid(neuron3)
    neuron1 += deltaHiddenSum1Neuron
    neuron2 += deltaHiddenSum2Neuron
    neuron3 += deltaHiddenSum3Neuron
    dW1 = deltaHiddenSum1Neuron / (input[0] * input[1])
    dW2 = deltaHiddenSum2Neuron / (input[0] * input[1])
    dW3 = deltaHiddenSum3Neuron / (input[0] * input[1])
    w1 += dW1
    w2 += dW2
    w3 += dW3
    w4 += dW1
    w5 += dW2
    w6 += dW3

print(output)
