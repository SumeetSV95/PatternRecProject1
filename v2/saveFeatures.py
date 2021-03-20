import os
import pickle

import numpy as np

import extactFeatures


def main():
    file_path = "C:\\Users\\Abhay\\Desktop\\pr_project_1\\trainingSymbols\\"
    UIToFeaturesDict = {}
    UIToLabelDict = {}
    size, file_counter = 0, 0
    for file in os.listdir(file_path):
        file_counter += 1
        if file.endswith(".inkml"):
            # add a parameter visualize and set it to true to see features
            UI, features = extactFeatures.getFeatures(file_path + file)
            if size == 0:
                size = features.shape[0]
            UIToFeaturesDict[UI] = features
            if file_counter % 100 == 0:
                print(file_counter)
    print("done creating features")
    with open(file_path + "iso_GT.txt") as reader:
        for line in reader:
            tempList = line.split(",")
            tempList[1] = tempList[1].replace("\\", "")
            UIToLabelDict[tempList[0]] = tempList[1].replace('\n', '')
    print("done reading label file")
    """
    featureArr = np.ndarray((0, size))
    UIArr = np.ndarray([])
    labelArr = np.array([])
    """
    featureArr, UIArr, labelArr = [], [], []
    i = 0
    for key in UIToFeaturesDict:
        """
        featureArr = np.append(featureArr, [UIToFeaturesDict[key]], axis=0)
        labelArr = np.append(labelArr, UIToLabelDict[key])
        UIArr = np.append(UIArr, key)
        """
        i += 1
        featureArr.append([UIToFeaturesDict[key]])
        labelArr.append(UIToLabelDict[key])
        UIArr.append(key)
        if i % 1000 == 0:
            print(i)
    print("created np arrays")
    featureFile = open("featureFile", "wb")
    pickle.dump([np.asarray(UIArr), np.asarray(featureArr), np.asarray(labelArr)], featureFile)
    featureFile.close()


if __name__ == '__main__':
    main()
