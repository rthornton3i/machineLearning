import numpy as np

import scrape_Functions as sf
import machineLearning_Functions as ml
import miscellaneous_Functions as misc

#https://www.python-course.eu/neural_networks_with_python_numpy.php

#stock = 'AAPL'
#url = 'https://finviz.com/quote.ashx?t=' + stock
#
#sf.scrape(url,stock)
#data = sf.readData('temp_' + stock + '.txt')
#
#searchIndex = 'Price'
#indexOffset = 4
#lineIndex = sf.lineSearch(searchIndex,data,indexOffset,exclusive=True)
#price = float(data[lineIndex].strip())
#
#searchIndex = 'P/E'
#indexOffset = 5
#lineIndex = sf.lineSearch(searchIndex,data,indexOffset,exclusive=True)
#pe = float(data[lineIndex].strip())

inputs = np.random.randint(-100,100,size=(1,4))
numHLs = [8,8]
numOuts = 2

[inputs,outputs,wts,nodes] = ml.networkSetup(inputs,numHLs,numOuts)