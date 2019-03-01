import numpy as np

# https://www.python-course.eu/neural_networks_with_python_numpy.php
# https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
# https://en.wikipedia.org/wiki/Backpropagation
# http://neuralnetworksanddeeplearning.com/chap2.html

class NeuralNetwork():
    
    def __init__(self,inputs,numOutputs,numHidNodes):
        """Initialize variables
            -"inputs" is an array of size (1,n) where n is the # of inputs
            -"numOutputs" is an array of [n] where n is the # of outputs
            -"numHidNodes" is an array of size (1,n) where n is the # of nodes in each hidden layer
        """
        
        self.inputs = np.reshape(inputs,(len(inputs),1))
        numInputs = [len(self.inputs)]
        
        self.numNodes = np.concatenate((numInputs,numHidNodes,numOutputs))
        self.numLayers = len(self.numNodes) - 1

    def layerSetup(self,symOpt):
        """Initialize network weights and biases
            -"symOpt" is True/False to specify symmetry about 0, either [-1,1] or (0,1], respectively
           
           Arrays of weights and biases are size (numOutputs,numInputs)
        """
        
        self.wts = [Misc.randArray((self.numNodes[n+1],self.numNodes[n]),symType=symOpt) for n in range(self.numLayers)] 
        self.bias = [Misc.randArray((self.numNodes[n+1],1),symType=symOpt) for n in range(self.numLayers)]
    
    def runNetwork(self,curveOpt,inputLims=None):
        """Run network given current weights and biases to calculate nodes/outputs
            -"curveOpt" is to specify type of normalizing curve
            -[OPT] "inputLims" is None or an array of [n,m] to specify lower and upper linear normalizing limits, respectively
        """
        
        self.inputs = self.inputs if inputLims is None else Misc.linNorm(self.inputs,lims=inputLims)
        
        nodes = []
        layerInputs = self.inputs    
        for n in range(self.numLayers):
            rawNode = np.dot(self.wts[n],layerInputs).reshape(self.numNodes[n+1],1) + self.bias[n]
            node = Misc.sigNorm(rawNode,curveType=curveOpt)
            
            nodes.append(node)
            
            layerInputs = node
        
        del(nodes[-1])
        self.outputs = node
        self.nodes = nodes
    
    def errorFunc(self,desOutputs,errorOpt='sqrd'):
        """Calculate network error based on desired outputs
            -"desOutputs" is an array of size (1,n) where n equals the number of inputs
        """
        
        desOutputs = np.reshape(desOutputs,(len(desOutputs),1))
        
        if errorOpt == 'sqrd':
            self.error = (1/2) * np.sum((self.outputs - desOutputs) ** 2)
        else:
            raise Exception('ERROR: Invalid error function specified.')
    
    def adjFunc(self,curveOpt):
        """Run network given current weights and biases to calculate nodes/outputs
            -"curveOpt" is to specify type of normalizing curve
            -[OPT] "inputLims" is None or an array of [n,m] to specify lower and upper linear normalizing limits, respectively
        """
        
        self.adjs = self.error * Misc.sigDeriv(self.outputs,curveType=curveOpt)
        
    def trainNetwork(self,trainData,epochs,learnRate,batchSize,curveOpt):
        """Train network based on training data and various parameters
            -"trainData" is an array of tuples of (inputs,desOutputs)
            -"epochs"
            -"learnRate"
            -"batchSize"
            -"curveOpt" is to specify type of normalizing curve
        """
        
        for n in range(epochs):
            miniBatch = [trainData[i:i+batchSize] for i in range(0, len(trainData), batchSize)]

            for batch in miniBatch:
                bias = [np.zeros(b.shape) for b in self.bias]
                wts = [np.zeros(w.shape) for w in self.wts]
        
                for inputs, desOutputs in batch:
                    tempBias = [np.zeros(b.shape) for b in self.bias]
                    tempWts = [np.zeros(w.shape) for w in self.wts]
            
                    # feedforward
                    self.runNetwork(curveOpt=None)
                    node = inputs
                    nodes = [inputs]
                    rawNodes = []
            
                    for b, w in zip(self.bias, self.wts):
                        rawNode = np.dot(w, node)+b
                        rawNodes.append(rawNode)
                        node = Misc.sigNorm(rawNode)
                        nodes.append(node)
                    
                    outputs = nodes[-1]
            
                    # backward pass
                    delta = (outputs - desOutputs) * Misc.sigDeriv(rawNodes[-1])
                    tempBias[-1] = delta
                    tempWts[-1] = np.dot(delta, nodes[-2].transpose())
            
                    for layer in range(2, self.numLayers):
                        rawNode = rawNodes[-layer]
                        nodeDeriv = Misc.sigDeriv(rawNode)
                        delta = np.dot(self.wts[-layer+1].transpose(), delta) * nodeDeriv
                        tempBias[-layer] = delta
                        tempWts[-layer] = np.dot(delta, nodes[-layer-1].transpose())
                        
                    bias = [nb+dnb for nb, dnb in zip(bias, tempBias)]
                    wts = [nw+dnw for nw, dnw in zip(wts, tempWts)]
        
                self.wts = [w-(learnRate/len(batch))*nw for w, nw in zip(self.wts, wts)]
                self.bias = [b-(learnRate/len(batch))*nb for b, nb in zip(self.bias, bias)]

class Misc():
    
    def linNorm(data,lims=[0,1]):
        minLim = lims[0]
        maxLim = lims[1]
        
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
        elif curveType == 'relu':
            normData = np.asarray([0 if data[n] < 0 else data[n] for n in range(len(data))]).reshape(len(data),1)
        elif curveType is None:
            normData = data
        else:
            raise Exception('ERROR: Invalid curve type specified.')
        
        return normData
        
    def randArray(dims,symType=False):
        randNums = np.random.random(size=dims)
        randNums = randNums * 2 - 1 if symType is True else randNums
        
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
        elif curveType == 'relu':
            derivData = np.asarray([0 if data[n] < 0 else 1 for n in range(len(data))]).reshape(len(data),1)
        else:
            raise Exception('ERROR: Invalid curve type specified.')
        
        return derivData


inputs = [1,5,8]
numHidNodes = [4,2]
numOutputs = [2]

desOutputs = [0,1]
normCurve = 'hyperbolic'

ntwk = NeuralNetwork(inputs,numOutputs,numHidNodes)
ntwk.layerSetup(symOpt=True)
ntwk.runNetwork(curveOpt=normCurve,inputLims=None)
#ntwk.errorFunc(desOutputs)
#ntwk.adjFunc(curveOpt=normCurve)

#for n in range(1):
#    [nodes,outputs] = runNetwork(inputs,self.wts,self.bias,lims=[-1,1],curveOpt='logistic')
#    error = errorFunc(outputs,desOutputs)
#    adjs = adjFunc(outputs,error,curveOpt='logistic')

print(ntwk.outputs)
print('')

activation = np.asarray(inputs).reshape(len(inputs),1)
for b, w in zip(ntwk.bias, ntwk.wts):
    z = np.dot(w, activation)+b
    activation = Misc.sigNorm(z)
    print(z)
    print(activation)
    print('')

ntwkAtt = vars(ntwk)