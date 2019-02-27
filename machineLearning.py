import numpy as np

# https://www.python-course.eu/neural_networks_with_python_numpy.php
# https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
# https://en.wikipedia.org/wiki/Backpropagation

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
        
        self.inputs = self.inputs if inputLims is None else Misc.linNorm(self.inputs,minLim=inputLims[0],maxLim=inputLims[1])
        
        nodes = []
        layerInputs = self.inputs    
        for n in range(self.numLayers):
            node = np.dot(self.wts[n],layerInputs).reshape(self.numNodes[n+1],1) + self.bias[n]
            node = Misc.sigNorm(node,curveType=curveOpt)
            
            nodes.append(node)
            
            layerInputs = node
        
        del(nodes[-1])
        self.outputs = node
        self.nodes = nodes
    
    def errorFunc(self,desOutputs):
        """Calculate network error based on desired outputs
            -"desOutputs" is an array of size (1,n) where n equals the number of inputs
        """
        
        desOutputs = np.reshape(desOutputs,(len(desOutputs),1))
        
        self.error = (1/2) * np.sum((self.outputs - desOutputs) ** 2)
    
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
                    activation = inputs
                    activations = [inputs] # list to store all the activations, layer by layer
                    zs = [] # list to store all the z vectors, layer by layer
            
                    for b, w in zip(self.bias, self.wts):
                        z = np.dot(w, activation)+b
                        zs.append(z)
                        activation = Misc.sigNorm(z)
                        activations.append(activation)
            
                    # backward pass
                    delta = (activations[-1] - desOutputs) * Misc.sigDeriv(zs[-1])
                    tempBias[-1] = delta
                    tempWts[-1] = np.dot(delta, activations[-2].transpose())
            
                    for l in range(2, self.numLayers):
                        z = zs[-l]
                        sp = Misc.sigDeriv(z)
                        delta = np.dot(self.wts[-l+1].transpose(), delta) * sp
                        tempBias[-l] = delta
                        tempWts[-l] = np.dot(delta, activations[-l-1].transpose())
                        
                    bias = [nb+dnb for nb, dnb in zip(bias, tempBias)]
                    wts = [nw+dnw for nw, dnw in zip(wts, tempWts)]
        
                self.wts = [w-(learnRate/len(batch))*nw for w, nw in zip(self.wts, wts)]
                self.bias = [b-(learnRate/len(batch))*nb for b, nb in zip(self.bias, bias)]

class Misc():
    
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