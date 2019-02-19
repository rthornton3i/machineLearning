import numpy as np

import miscellaneous_Functions as misc

def layerSetup(numInputs,numOutputs,layers,symOpt,curvOpt):    
    networkWts = []
    networkBias = []
    
    layers = np.concatenate((numInputs,layers,numOutputs))
    numLayers = len(layers) - 1
    
    for n in range(numLayers):
        wts = misc.randArray((layers[n+1],numInputs),sym=symOpt)
        bias = misc.randArray((numOutputs,1),sym=symOpt)
        
        networkWts.append(wts)
        networkBias.append(bias)
    
    return [inputs,wts,bias,outputs]

def runNetwork():
    inputs = misc.linNorm(inputs)   
    outputs = np.dot(wts,inputs).reshape(numOutputs,1) + bias
    outputs = misc.sigNorm(outputs,curveType=curvOpt)
    
def networkSetup(inputs,numNodes,numOutputs,symOpt,curvOpt):
    nodes = np.concatenate((numNodes,numOutputs))
    numLayers = len(nodes)
    
    networkNodes = []
    networkWts = []
    networkBias = []
    
    for n in range(numLayers):
        layerInputs = inputs
            
        [layerInputs,layerWts,layerBias,layerOutputs] = layerSetup(layerInputs,nodes[n],symOpt,curvOpt)
        networkNodes.append(layerInputs)
        networkWts.append(layerWts)
        networkBias.append(layerBias)
        
        inputs = layerOutputs
        networkOutputs = layerOutputs

    return [networkNodes,networkWts,networkBias,networkOutputs]

def costFunc(outputs,desiredOutputs):
    desiredOutputs = np.reshape(desiredOutputs,(len(desiredOutputs),1))
    
    cost = np.sum((outputs - desiredOutputs) ** 2)
    
    return cost

def networkAdjust(inputs,wts,bias):

    
    return [inputs,allWts,allNodes]


inputs = [2,5,9,10]

numNodes = [8,6]
numOutputs = [3]

desiredOutputs = [0,0,1]

[nodes,wts,bias,outputs] = networkSetup(inputs,numNodes,numOutputs,symOpt=True,curvOpt='logistic')
cost = costFunc(outputs,desiredOutputs)
    
