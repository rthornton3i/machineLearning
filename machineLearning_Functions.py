import numpy as np

import miscellaneous_Functions as misc

def layerSetup(numInputs,numOutputs,layers,symOpt):    
    layers = np.concatenate((numInputs,layers,numOutputs))
    numLayers = len(layers) - 1
    
    wts = [misc.randArray((layers[n+1],layers[n]),sym=symOpt) for n in range(numLayers)] 
    bias = [misc.randArray((layers[n+1],1),sym=symOpt) for n in range(numLayers)]
    
    return [wts,bias]

def runNetwork(inputs,wts,bias,lims,curveOpt):
    nodes = []
    
    inputs = np.reshape(misc.linNorm(inputs,minLim=lims[0],maxLim=lims[1]),(len(inputs),1))
    numLayers = len(wts)
    
    layerInputs = inputs    
    for n in range(numLayers):
        numOutputs = np.shape(wts[n])[0]
        node = np.dot(wts[n],layerInputs).reshape(numOutputs,1) + bias[n]
        node = misc.sigNorm(node,curveType=curveOpt)
        
        nodes.append(node)
        
        layerInputs = node
    
    outputs = node
    del(networkNodes[-1])
    
    return [nodes,outputs]

def errorFunc(outputs,desiredOutputs):
    desiredOutputs = np.reshape(desiredOutputs,(len(desiredOutputs),1))
    
    error = np.sum((outputs - desiredOutputs) ** 2)
    
    return error

def adjFunc(outputs,error,curveOpt):
    adjs = error * misc.sigDeriv(outputs,curveType=curveOpt)
    
    return adjs
    
def adjNetwork(adjs,):
    pass

numInputs = [5]
layers = [8,6]
numOutputs = [3]

inputs = [1,5,8,9,6]
desiredOutputs = [0,0,1]

[wts,bias] = layerSetup(numInputs,numOutputs,layers,symOpt=True)

for n in range(1):
    [nodes,outputs] = runNetwork(inputs,wts,bias,lims=[-1,1],curveOpt='logistic')
    error = errorFunc(outputs,desiredOutputs)
    adjs = adjFunc(outputs,error,curveOpt='logistic')
    
    