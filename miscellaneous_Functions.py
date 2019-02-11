import numpy as np

def linNorm(data,minLim,maxLim):
    dataMax = np.max(data)
    dataMin = np.min(data)
    
    normData = minLim + (((data - dataMin)/(dataMax - dataMin)) * (maxLim - minLim))
    
    return normData

def sigNorm(data):
    normData = 1 / (1 + np.exp(-data))
    
    return normData
