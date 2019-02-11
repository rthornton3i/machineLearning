import numpy as np

import miscellaneous_Functions as misc

def networkSetup(inputs,numHLs,numOuts,minLim=0,maxLim=1):
    inputs = np.asarray(inputs)
    ht,wd = np.shape(inputs)
    
    inputs = np.reshape(inputs,(wd,1)) if ht == 1 else inputs
    ht,wd = np.shape(inputs)
    
    inputs = misc.linNorm(inputs,minLim,maxLim)
    
    allWts = []
    allNodes = []
    numLoops = len(numHLs)+1
    for n in range(numLoops):
        if n == 0:
            wts = np.random.rand(numHLs[n],ht)
            nodes = misc.sigNorm(np.dot(wts,inputs))
        elif n == numLoops-1:
            wts = np.random.rand(numOuts,numHLs[n-1])
            nodes = misc.sigNorm(np.dot(wts,allNodes[n-1]))
        else:
            wts = np.random.rand(numHLs[n],numHLs[n-1])
            nodes = misc.sigNorm(np.dot(wts,allNodes[n-1]))
        
        allWts.append(wts)
        allNodes.append(nodes)
        
    return [inputs,allWts,allNodes]

def networkTrain(inputs,allWts,allNodes):
    
    
    return [inputs,allWts,allNodes]
