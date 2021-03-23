import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from skimage.feature import hog


def generateFeatures(strokes, filename, DPI):
    # Plots figure
    plt.figure(num=None, figsize=(1, 1), dpi=DPI)
    for trace in strokes:
        for i in range(0, trace.shape[0] - 1):
            plt.plot((trace[i:i + 2, 0]), trace[i:i + 2, 1], color='black')
    plt.axis('equal')
    plt.axis('off')

    name = filename + '.png'
    plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    image = cv.imread(name)
    image = cv.flip(image, 0)
    image = cv.bitwise_not(image)

    cv.imwrite(name, image)
    img = Image.open(name)
    imArr = np.asarray(img)

    fg = hog(imArr, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1), block_norm="L2")
    os.remove(name)

    return fg


def getFeatures(fileName, DPI):
    with open(fileName) as fp:
        contents = fp.read()
        soup = BeautifulSoup(contents, 'lxml')
        annotations = soup.find_all("annotation", attrs={"type": "UI"})
        UI = annotations[0].string
        traces = soup.find_all("trace")
        strokes = []  # List of numpy arrays
        for trace in traces:
            list_of_trace = trace.string.strip().split(",")
            temp_list = []
            for element in list_of_trace:
                clean_element = element.strip().split(" ")
                temp_list.append([float(x) for x in clean_element])
            strokes.append(np.asarray(temp_list))
        features = generateFeatures(strokes, fileName, DPI)
        return UI, features
