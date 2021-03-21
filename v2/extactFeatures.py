import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from skimage.feature import hog


def generateFeatures(strokes):
    # Plots figure
    fig = plt.figure(num=None, figsize=(1, 1), dpi=70)
    for trace in strokes:
        plt.plot(trace[:, 0], trace[:, 1], color='black')
    plt.axis('equal')
    plt.axis('off')

    # name = filename + '.png'
    # plt.savefig(name, bbox_inches='tight', pad_inches=0.05)

    # New part
    plt.margins(0.05, tight=True)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    """
    image = cv.imread(name)  # image to nd-array RGB
    image = cv.flip(image, 0)
    image = cv.bitwise_not(image)  # no clue

    cv.imwrite(name, image)
    img = Image.open(name)
    imArr = np.asarray(img)
    """
    image = cv.flip(data, 0)
    image = cv.bitwise_not(image)

    """
    if visualise:
        fg, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),
                            block_norm="L2", visualize=True)
        hogImage = Image.fromarray(np.uint8(hog_image)).convert('RGB')
        hogImage.save(filename + "feature.png")
    else:
        fg = hog(image, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),
                 block_norm="L2")
        os.remove(name)
    """
    fg = hog(image, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1), block_norm="L2")
    return fg


def getFeatures(fileName):
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
        features = generateFeatures(strokes)
        return UI, features
