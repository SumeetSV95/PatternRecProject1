import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
from bs4 import BeautifulSoup
from bs4 import element as ele
from PIL import Image
import cv2 as cv
import os
from skimage.feature import hog
from skimage import data, exposure


class Symbol:
    def __init__(self):
        self.traces = None
        self.classification = None
        self.UI = None


def getDistance(p1, p2):
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def getCentroid(traces):
    sumMixi = 0
    sumM = 0
    sumMiyi = 0
    for trace in traces:
        for i in range(0, trace.shape[0] - 1):
            mass = getDistance(trace[i], trace[i + 1])
            sumM += mass
            sumMixi += mass * (trace[i + 1][0] + trace[i][0]) / 2
            sumMiyi += mass * (trace[i + 1][1] + trace[i][1]) / 2
    return sumMixi / sumM, sumMiyi / sumM


def generatePlot(traces, j):
    plt.figure(num=None, figsize=(1, 1), dpi=70)
    for trace in traces:

        for i in range(0, trace.shape[0] - 1):
            plt.plot((trace[i:i + 2, 0]), trace[i:i + 2, 1], color='black')
    # x, y = getCentroid(traces)
    # plt.ylim(y-25, y+25)
    # plt.xlim(x - 25, x + 25)
    plt.axis('equal')
    plt.axis('off')
    name = 'test' + str(j) + '.png'

    plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    image = cv.imread(name)
    image = cv.flip(image, 0)
    image = cv.bitwise_not(image)

    img = cv.GaussianBlur(image, (5, 5), 1)
    temp = np.zeros((50, 50))
    finalImg = cv.normalize(img, temp, 0, 255, cv.NORM_MINMAX)

    cv.imwrite(name, finalImg)


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
    for i, symbol in enumerate(symbolObjs):
        print(symbol.UI)
        print(symbol.traces)
        generatePlot(symbol.traces, i)
    for file in os.listdir("/Users/sumeet95/PycharmProjects/pythonProject1"):
        print(file)
        if file.endswith(".png"):

            img = Image.open(file)
            imArr = np.asarray(img)
            fg, hog_image = hog(imArr, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),
                                block_norm="L2",
                                visualize=True, )
            print(fg.shape)
            hogImage = Image.fromarray(np.uint8(hog_image)).convert('RGB')
            hogImage.save("feature_"+str(file))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
