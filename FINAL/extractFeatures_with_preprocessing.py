import math
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
from PIL import Image
from bs4 import BeautifulSoup
from skimage.feature import hog


def slope(p1, p2):
    """
    Slope between 2 points
    :param p1: point 1
    :param p2: point 2
    :return: slope
    """
    if p1[0] == p2[0]:
        return float('inf')
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def getAngle(points):
    """
    Angle
    :param points:
    :return:
    """
    pt1, pt2, pt3 = points[0], points[1], points[2]
    angle = abs((slope(pt2, pt1)) - (slope(pt3, pt2)))
    return math.degrees(angle) % 360


def get_diagonal_line(trace):
    """
    Finds the diagonal line between the points
    :param trace: pair of points
    :return: Ld
    """
    x, y = trace[:, 0], trace[:, 1]

    minX, maxX = np.min(x), np.max(x)
    minY, maxY = np.min(y), np.max(y)

    line = math.sqrt(math.pow(maxX - minX, 2) + math.pow(maxY - minY, 2))
    return line


def get_hook_params(sharpPoints):
    """
    Len and angle
    :param sharpPoints: 3 points
    :return: [params]
    """
    hook_len = np.linalg.norm(sharpPoints[0] - sharpPoints[1])
    angle = getAngle(sharpPoints[0:3])
    return [hook_len, angle]


def hook_condition(params, threshold_length):
    """
    Hook length <= threshold
    Angle not perpendicular
    :param params: hook parameters
    :param threshold_length:
    :return: Bool
    """
    if params[0] <= threshold_length and params[1] <= 90:
        return True
    return False


def get_hooks(sharp_points, trace):
    """
    Finds hooks
    :param sharp_points:
    :param trace:
    :return: List of hooks
    """
    if len(sharp_points) <= 2:
        return sharp_points
    total_len = len(sharp_points)
    # ---------------------------------------------------------------------------------------------------
    threshold_length = get_diagonal_line(trace) * 0.05
    # ---------------------------------------------------------------------------------------------------
    start_params = get_hook_params(sharp_points)

    end_points = sharp_points[-3:]
    end_points = end_points[::-1]

    end_params = get_hook_params(end_points)
    # ---------------------------------------------------------------------------------------------------
    if hook_condition(end_params, threshold_length):
        sharp_points = np.delete(sharp_points, [total_len - 1], 0)

    total_len = len(sharp_points)

    if hook_condition(start_params, threshold_length) and total_len > 2:
        sharp_points = np.delete(sharp_points, [0], 0)
    return sharp_points


def get_cosine(points):
    """
    3 points -> Line cosine
    :param points: list of 3 points
    :return: Angle
    """
    p1, p2, p3 = points[0], points[1], points[2]

    # Sanity check (make sure only 2 "traces")
    v1 = np.asarray([p1[0], p1[1]])
    v2 = np.asarray([p2[0], p2[1]])
    v3 = np.asarray([p3[0], p3[1]])

    line1 = v1 - v2
    line2 = v3 - v2

    dotProd = line1 @ line2.T

    magLine1 = np.linalg.norm(line1)
    magLine2 = np.linalg.norm(line2)
    # Cosine
    if (magLine2 * magLine1) == 0:
        return 20  # Placeholder impossible cosing angle
    cos = dotProd / (magLine2 * magLine1)
    cos = round(cos, 12)
    return math.cos(cos)


def sharpPointDetection(trace):
    """
    Finds sharp points
    :param trace: List of all points
    :return: Np array of sharp points
    """
    list_of_sharp_points, angles = [], []
    # ---------------------------------------------------------------------------------------------------
    # Possible angles
    for i in range(len(trace) - 2):
        angle = get_cosine(trace[i:i + 3])
        if angle == 20:  # Placeholder - impossible value
            continue
        angles.append(angle)
    # ---------------------------------------------------------------------------------------------------
    # List of sharp points
    list_of_sharp_points.append(trace[0])
    theta_prev = 0
    for i in range(len(angles) - 1):
        theta = angles[i] - angles[i + 1]
        if theta == 0:
            theta_prev = theta
            continue
        delta = theta * theta_prev
        if delta <= 0 and theta_prev != 0:
            list_of_sharp_points.append(trace[i + 1])
        theta_prev = theta
    list_of_sharp_points.append(trace[len(trace) - 1])
    # ---------------------------------------------------------------------------------------------------
    return np.asarray(list_of_sharp_points)


def generateFeatures(strokes, filename, DPI):
    """
    Generate features
    :param strokes: inkml points
    :param filename: name of the file
    :param DPI: resolution of the generated image
    :return: features
    """
    plt.figure(num=None, figsize=(1, 1), dpi=DPI)
    # Smooth the points one-by-one
    for trace in strokes:
        sharpPoints = sharpPointDetection(trace)
        sharpPoints = get_hooks(sharpPoints, trace)
        # ---------------------------------------------------------------------------------------------------
        x, y = sharpPoints[:, 0], sharpPoints[:, 1]

        n = x.shape[0]
        plot_points = 100

        # Parametric spline representation
        k = 3 if n > 3 else n - 1

        knot_space = range(n)
        # ---------------------------------------------------------------------------------------------------
        # find knot vector at-most 3 points
        knots = si.InterpolatedUnivariateSpline(knot_space, knot_space, k=k).get_knots()
        knots_full = np.concatenate(([knots[0]] * k, knots, [knots[-1]] * k))

        # define tuples of knot vector, coefficient vector (control points coordinates) and spline degree
        tckX = knots_full, x, k
        tckY = knots_full, y, k

        # construct spline functions
        splineX = si.UnivariateSpline._from_tck(tckX)
        splineY = si.UnivariateSpline._from_tck(tckY)

        # evaluate spline points
        tP = np.linspace(knot_space[0], knot_space[-1], plot_points)
        xP, yP = splineX(tP), splineY(tP)
        # ---------------------------------------------------------------------------------------------------
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        offset_x = (x_max - x_min) * 0.05
        offset_y = (y_max - y_min) * 0.1

        plt.plot(xP, yP, 'b', lw=2.5, color='black')

        if offset_x == 0:
            offset_x += 0.001
        if offset_y == 0:
            offset_y += 0.001

        plt.xlim([x_min - offset_x, x_max + offset_x])
        plt.ylim([y_min - offset_y, y_max + offset_y])
    # ---------------------------------------------------------------------------------------------------
    plt.axis('equal')
    plt.axis('off')

    name = filename + '.png'
    plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    image = cv.imread(name)
    image = cv.flip(image, 0)
    image = cv.bitwise_not(image)    # To create MNIST-type image (white on black bg)

    cv.imwrite(name, image)
    img = Image.open(name)
    imArr = np.asarray(img)
    # ---------------------------------------------------------------------------------------------------
    fg = hog(imArr, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),
             block_norm="L2")
    os.remove(name)
    return fg


def remove_duplicates(current_list):
    """
    De-duplication of duplicate consecutive points
    :param current_list:
    :return:
    """
    t1 = [v for i, v in enumerate(current_list) if i == 0 or v != current_list[i - 1]]
    return t1


def getFeatures(fileName, DPI):
    """
    Gets features from the inkml files
    :param DPI: Resolution of plotted image
    :param fileName: Filename
    :return:
    """
    with open(fileName) as fp:
        contents = fp.read()
        soup = BeautifulSoup(contents, 'lxml')
        annotations = soup.find_all("annotation", attrs={"type": "UI"})
        UI = annotations[0].string
        traces = soup.find_all("trace")
        strokes = []  # List of np_array

        for trace in traces:
            list_of_trace = trace.string.strip().split(",")
            temp_list = []
            for element in list_of_trace:
                temp = element.strip().split(" ")
                temp_list.append([float(x) for x in temp])
            de_duplication = remove_duplicates(temp_list)
            np_temp = np.asarray(de_duplication)
            if np_temp.shape[1] == 3:
                np_temp = np.delete(np_temp, 2, 1)
            strokes.append(np_temp)
        all_features = generateFeatures(strokes, fileName, DPI)
        return UI, all_features
