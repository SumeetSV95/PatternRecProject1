import os

from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from skimage.feature import hog
import scipy.interpolate as si
import math


def getAngle(pt1, pt2, pt3):
    angle = abs(((pt2[1] - pt1[1]) / (pt2[0] - pt1[0])) - (
            (pt3[1] - pt2[1]) / (pt3[0] - pt2[0])))

    return math.degrees(angle) % 360


def getDistance(p1, p2):
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def addPoint(traceLis, i, d):
    xi, yi = traceLis[i]
    xi1, yi1 = traceLis[i + 1]
    if xi == xi1:
        i += 1
        return i
    k = (yi1 - yi) / (xi1 - xi)
    if xi1 > xi:
        xPrime = xi + math.sqrt((d * d) / (k * k + 1))
    else:
        xPrime = xi - math.sqrt((d * d) / (k * k + 1))
    if yi1 > yi:
        yPrime = yi + d
    elif yi1 < yi:
        yPrime = yi - d
    else:
        yPrime = k * xPrime + yi - k * xi
    traceLis.insert(i + 1, [xPrime, yPrime])

    i += 2
    return i


def sharpPointDetection(trace):
    sharpList = []
    angles = []
    for i in range(0, len(trace) - 2):
        p1 = trace[i]
        p2 = trace[i + 1]
        p3 = trace[i + 2]
        v1 = np.asarray([p1[0], p1[1]])
        v2 = np.asarray([p2[0], p2[1]])
        v3 = np.asarray([p3[0], p3[1]])
        line1 = v1 - v2
        line2 = v3 - v2

        dotProd = line1 @ line2.T

        magLine1 = np.linalg.norm(line1)
        magLine2 = np.linalg.norm(line2)
        cos = dotProd / (magLine2 * magLine1)
        cos = round(cos, 12)

        angle = math.acos(cos)
        angles.append(angle)

    sharpList.append(trace[0])
    thetaPrev = 0
    for i in range(0, len(angles) - 1):
        alpha_i = angles[i]
        alpha_i_prime = angles[i + 1]
        theta = alpha_i - alpha_i_prime
        if theta == 0:
            thetaPrev = theta
            continue
        delta = theta * thetaPrev
        if delta <= 0 and thetaPrev != 0:
            sharpList.append(trace[i + 1])
        thetaPrev = theta
    sharpList.append(trace[len(trace) - 1])
    return sharpList


def generateFeatures(strokes, filename, visualise):
    def interpolatePoints(traces):
        strokeList = []
        for trace in traces:
            print(trace)
            total = 0
            numberOfFeatures = trace.shape[0]

            for i in range(0, numberOfFeatures - 1):
                dist = np.linalg.norm(trace[i] - trace[i + 1])
                total += dist

            avg = total / numberOfFeatures
            d = avg*0.5

            traceLis = trace.tolist()
            i = 0
            while i < len(traceLis) - 1:

                xi, yi = traceLis[i]
                xi1, yi1 = traceLis[i + 1]
                dist = getDistance([xi, yi], [xi1, yi1])

                if dist >= d:
                    i = addPoint(traceLis, i, d)
                else:
                    total = 0
                    n = i
                    while n + 1 < len(traceLis) and total < d:
                        total += getDistance(traceLis[n], traceLis[n + 1])
                        n += 1
                    checkPoint = traceLis[n - 2]
                    added = False
                    if n < len(traceLis) and getDistance(checkPoint, traceLis[n - 1]) >= d:
                        index = addPoint(traceLis, n - 2, d)
                        if index != n - 2:
                            added = True
                    del traceLis[i + 1:n - 1]

                    if added:
                        i += 2
                    else:
                        i += 1

            strokeList.append(np.asarray(traceLis))
        return np.asarray(strokeList)

    # strokes = interpolatePoints(strokes)
    plt.figure(num=None, figsize=(1, 1), dpi=50)
    for trace in strokes:
        sharpPoints = sharpPointDetection(trace)
        sharpPoints = np.asarray(sharpPoints)
        sharpPoints = hookDetection(sharpPoints, trace)
        x = sharpPoints[:, 0]
        y = sharpPoints[:, 1]

        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        n = x.shape[0]

        plotpoints = 100

        #####  PARAMETRIC SPLINE REPRESENTATION  #####
        if n <= 3:
            k = n - 1
        else:
            k = 3
        knotspace = range(n)

        # find knot vector
        knots = si.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
        knots_full = np.concatenate(([knots[0]] * k, knots, [knots[-1]] * k))

        # define tuples of knot vector, coefficient vector (control points coordinates) and spline degree
        tckX = knots_full, x, k
        tckY = knots_full, y, k

        # construct spline functions
        splineX = si.UnivariateSpline._from_tck(tckX)
        splineY = si.UnivariateSpline._from_tck(tckY)

        # evaluate spline points
        tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)
        xP = splineX(tP)
        yP = splineY(tP)
        offset_x = (xmax - xmin) * 0.05
        offset_y = (ymax - ymin) * 0.1
        plt.plot(xP, yP, 'b', lw=2.5, color='black')
        plt.xlim([xmin - offset_x, xmax + offset_x])
        plt.ylim([ymin - offset_y, ymax + offset_y])
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

    # img = cv.GaussianBlur(image, (3, 3), 1)
    # temp = np.zeros((50, 50))
    # finalImg = cv.normalize(img, temp, 0, 255, cv.NORM_MINMAX)

    cv.imwrite(name, image)
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


def hookDetection(sharpPoints, trace):
    if len(sharpPoints) <= 2:
        return sharpPoints
    x = trace[:, 0]
    y = trace[:, 1]
    minX, maxX = np.min(x), np.max(x)
    minY, maxY = np.min(y), np.max(y)
    thresholdLen = (math.sqrt(math.pow(maxX - minX, 2) + math.pow(maxY - minY, 2))) * 0.05

    l = len(sharpPoints)
    pStartpt1 = sharpPoints[0]
    pStartpt2 = sharpPoints[1]
    pStartpt3 = sharpPoints[2]
    lenStartHook = np.linalg.norm(pStartpt1 - pStartpt2)

    startAngle = getAngle(pStartpt1, pStartpt2, pStartpt3)
    pEndpt1 = sharpPoints[l - 1]
    pEndpt2 = sharpPoints[l - 2]
    pEndpt3 = sharpPoints[l - 3]
    lenEndHook = np.linalg.norm(pEndpt1 - pEndpt2)
    endAngle = getAngle(pEndpt1, pEndpt2, pEndpt3)
    if lenEndHook <= thresholdLen and endAngle <= 90:
        sharpPoints = np.delete(sharpPoints, [l - 1], 0)
    l = len(sharpPoints)
    if lenStartHook <= thresholdLen and startAngle <= 90 and l > 2:
        sharpPoints = np.delete(sharpPoints, [0], 0)

    return sharpPoints


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
            newLis = np.asarray(newLis)
            if newLis.shape[1] == 3:
                newLis = np.delete(newLis, 2, 1)
            strokes.append(newLis)
        features = generateFeatures(strokes, fileName, visualize)
        return UI, features
