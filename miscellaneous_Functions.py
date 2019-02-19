import numpy as np
import matplotlib.pyplot as plt

def linNorm(data,minLim=0,maxLim=1):
    dataMax = np.max(data)
    dataMin = np.min(data)
    
    normData = minLim + (((data - dataMin)/(dataMax - dataMin)) * (maxLim - minLim))
    
    return normData

def sigNorm(data,curveType='logistic'):
    if curveType == 'logistic':
        normData = 1 / (1 + np.exp(-data))
    elif curveType == 'hyperbolic':
        normData = np.tanh(data)
    elif curveType == 'arctan':
        normData = np.arctan(data)
    elif curveType == 'sqrt':
        normData = data / np.sqrt(1 + data ** 2)
    else:
        raise Exception('ERROR: Invalid curve type specified.')
    
    return normData
    
def randArray(dims,sym=False):
    randNums = np.random.random(size=dims)
    randNums = randNums * 2 - 1 if sym is True else randNums
    
    return randNums

def sigDeriv(data,curveType='logistic'):
    if curveType == 'logistic':
        derivData = np.exp(data) / ((1 + np.exp(data)) ** 2)
    elif curveType == 'hyperbolic':
        derivData = 1 / (np.cosh(data)**2)
    elif curveType == 'arctan':
        derivData = 1 / (1 + data ** 2)
    elif curveType == 'sqrt':
        derivData = 1 / ((1 + data ** 2) ** (3/2))
    else:
        raise Exception('ERROR: Invalid curve type specified.')
    
    return derivData