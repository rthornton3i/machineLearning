import numpy as np

import miscellaneous_Functions as misc

def layerSetup(inputs,numOutputs,symOpt,curvOpt):
    inputs = np.reshape(inputs,(np.shape(inputs)[1],1)) if np.shape(inputs)[0] == 1 else inputs
    inputs = misc.linNorm(inputs)   
    numInputs = np.shape(inputs)[0]
    
    wts = misc.randArray((numOutputs,numInputs),sym=symOpt)
    bias = misc.randArray((numOutputs,1),sym=symOpt)
    
    outputs = np.dot(wts,inputs).reshape(numOutputs,1) + bias
    outputs = misc.sigNorm(outputs,curveType=curvOpt)
    
    return [inputs,wts,bias,outputs]

def networkSetup(inputs,numNodes,symOpt=False,curvOpt='logistic'):
    numLayers = len(numNodes)
    networkNodes = []
    networkWts = []
    networkBias = []
    
    for n in range(numLayers):
        layerInputs = inputs
            
        [layerInputs,layerWts,layerBias,layerOutputs] = layerSetup(layerInputs,numNodes[n],symOpt,curvOpt)
        networkNodes.append(layerInputs)
        networkWts.append(layerWts)
        networkBias.append(layerBias)
        
        inputs = layerOutputs
        networkOutputs = layerOutputs

    return [networkNodes,networkWts,networkBias,networkOutputs]

inputs = [2,5,9,10]
numNodes = [8,6,2]

[nodes,wts,bias,outputs] = networkSetup(inputs,numNodes,symOpt=True,curvOpt='hyperbolic')

print(outputs)

#def networkSetup(inputs,numHLs,numOuts,minLim=0,maxLim=1):
#    inputs = np.asarray(inputs)
#    ht,wd = np.shape(inputs)
#    
#    inputs = np.reshape(inputs,(wd,1)) if ht == 1 else inputs
#    ht,wd = np.shape(inputs)
#    
#    inputs = misc.linNorm(inputs,minLim,maxLim)
#    
#    allWts = []
#    allNodes = []
#    numLoops = len(numHLs)+1
#    for n in range(numLoops):
#        if n == 0:
#            wts = np.random.rand(numHLs[n],ht)
#            nodes = misc.sigNorm(np.dot(wts,inputs))
#        elif n == numLoops-1:
#            wts = np.random.rand(numOuts,numHLs[n-1])
#            nodes = misc.sigNorm(np.dot(wts,allNodes[n-1]))
#        else:
#            wts = np.random.rand(numHLs[n],numHLs[n-1])
#            nodes = misc.sigNorm(np.dot(wts,allNodes[n-1]))
#        
#        allWts.append(wts)
#        allNodes.append(nodes)
#        
#    outputs = allNodes[-1]
#    del(allNodes[-1])
#        
#    return [inputs,outputs,allWts,allNodes]

def networkTrain(inputs,allWts,allNodes):
    
    
    return [inputs,allWts,allNodes]


