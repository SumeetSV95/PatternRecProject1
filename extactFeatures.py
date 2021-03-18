import os

from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from skimage.feature import hog


def generateFeatures(strokes, filename, visualise):
    plt.figure(num=None, figsize=(1, 1), dpi=70)
    for trace in strokes:

        for i in range(0, trace.shape[0] - 1):
            plt.plot((trace[i:i + 2, 0]), trace[i:i + 2, 1], color='black')
    # x, y = getCentroid(traces)
    # plt.ylim(y-25, y+25)
    # plt.xlim(x - 25, x + 25)
    plt.axis('equal')
    plt.axis('off')
    name = filename + '.png'
    plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    image = cv.imread(name)
    image = cv.flip(image, 0)
    image = cv.bitwise_not(image)

    img = cv.GaussianBlur(image, (3, 3), 1)
    temp = np.zeros((50, 50))
    finalImg = cv.normalize(img, temp, 0, 255, cv.NORM_MINMAX)

    cv.imwrite(name, finalImg)
    img = Image.open(name)
    imArr = np.asarray(img)
    if visualise:
        fg, hog_image = hog(imArr, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),
                            block_norm="L2", visualize=True)
        hogImage = Image.fromarray(np.uint8(hog_image)).convert('RGB')
        hogImage.save(filename + "feature.png")
    else:
        fg = hog(imArr, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),
                 block_norm="L2")
        os.remove(name)
    return fg


def getFeatures(fileName, visualize=None):
    with open(fileName) as fp:
        contents = fp.read()
        soup = BeautifulSoup(contents, 'lxml')

        annotations = soup.find_all("annotation", attrs={"type": "UI"})
        UI = annotations[0].string
        traces = soup.find_all("trace")
        strokes = []
        for trace in traces:
            lis = trace.string.strip().split(",")
            newLis = []
            for ele in lis:
                newLis.append(list(map(float, ele.strip().split(" "))))
            strokes.append(np.asarray(newLis))
        features = generateFeatures(strokes, fileName, visualize)
        return UI, features
