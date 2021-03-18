import numpy as np

import os

import extactFeatures
import pickle


def main():
    filePath = "/Users/sumeet95/Downloads/trainingSymbols/"
    UIToFeaturesDict = {}
    UIToLabelDict = {}
    size = 0
    filesDone = 0
    for file in os.listdir(filePath):
        filesDone += 1
        if file.endswith(".inkml"):
            UI, features = extactFeatures.getFeatures(filePath + file)
            if size == 0:
                size = features.shape[0]
            UIToFeaturesDict[UI] = features
            if filesDone % 100 == 0:
                print(filesDone)

    with open(filePath + "iso_GT.txt") as reader:

        for line in reader:
            tempList = line.split(",")
            tempList[1] = tempList[1].replace("\\", "")
            UIToLabelDict[tempList[0]] = tempList[1].replace('\n', '')
    featureArr = np.ndarray((0, size))
    UIArr = np.ndarray([])
    labelArr = np.array([])
    for key in UIToFeaturesDict:
        featureArr = np.append(featureArr, [UIToFeaturesDict[key]], axis=0)
        labelArr = np.append(labelArr, UIToLabelDict[key])
        UIArr = np.append(UIArr, key)

    featureFile = open("featureFile", "wb")
    pickle.dump([UIArr, featureArr, labelArr], featureFile)
    featureFile.close()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
