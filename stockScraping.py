from urllib.request import urlopen
from bs4 import BeautifulSoup as bs

# https://www.macrotrends.net/stocks/charts/NFLX/netflix/stock-price-history

def scrape(url,name):
    html = urlopen(url)
    
    soup = bs(html, 'lxml')
    code = soup.prettify()

    fileName = 'temp_' + name + '.txt'
    
    try:
        file = open(fileName,'w+')
        file.write(code)    
    except UnicodeEncodeError:
        file = open(fileName,'w+',encoding="utf-8")
        file.write(code)  
        
    file.close()    

def readData(fileName):
    file = open(fileName,'r')
    data = file.readlines()
    file.close()    
    
    return data

def lineSearch(searchIndex,data,indexOffset,exclusive=True):
    lineIndex = -1
    
    for line in data:
        lineIndex += 1
        if exclusive is True:
            if searchIndex == line.strip():
                searchResult = True
                break
            else:
                searchResult = False
        else:
            if searchIndex in line.strip():
                searchResult = True
                break
            else:
                searchResult = False
    
    lineIndex += indexOffset
    
    if searchResult is False:
        lineIndex = None
    
    return lineIndex
    
def attSearch(searchIndex,data,index):
    lineIndex = lineSearch(searchIndex,data,index)

    try:       
        att = float(data[lineIndex].strip().replace(',',''))
    except ValueError:
        att = data[lineIndex].strip().replace(',','')
        if '<!DOCTYPE html>' in att or 'N/A' in att.upper():
            att = 'N/a'
    
    return att

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