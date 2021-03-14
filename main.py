import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import sys
from bs4 import BeautifulSoup
from bs4 import element as ele


class Symbol:
    def __init__(self):
        self.traces = None
        self.classification = None
        self.UI = None


def generatePlot(traces):
    for trace in traces:

        for i in range(0, trace.shape[0] - 1):
            plt.plot((trace[i:i + 2, 0]), trace[i:i + 2, 1], color='black')
    plt.axis('equal')
    plt.show()


def generatePath(filePath, i):
    return filePath + "iso{}.inkml".format(i)


def main():
    filePath = "/Users/sumeet95/Downloads/trainingSymbols/"
    symbolObjs = []
    for i in range(10):
        newFilePath = generatePath(filePath, i)
        newSymbol = Symbol()
        with open(newFilePath) as fp:
            contents = fp.read()
            soup = BeautifulSoup(contents, 'lxml')

            annotations = soup.find_all("annotation", attrs={"type": "UI"})
            UI = annotations[0].string
            newSymbol.UI = UI
            traces = soup.find_all("trace")
            strokes = []
            for trace in traces:
                lis = trace.string.strip().split(",")
                newLis = []
                for ele in lis:
                    newLis.append(list(map(int, ele.strip().split(" "))))
                strokes.append(np.asarray(newLis))
            newSymbol.traces = strokes
            symbolObjs.append(newSymbol)
    for symbol in symbolObjs:
        print(symbol.UI)
        print(symbol.traces)
        generatePlot(symbol.traces)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
