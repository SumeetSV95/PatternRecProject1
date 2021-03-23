import os
import pickle
import time
import numpy as np
import datetime
import extractFeatures
import extractFeatures_with_preprocessing


def main():
    # Pass these parameters
    DPI, f = 70, 1296
    preprocess = True
    file_path = "C:\\Users\\Abhay\\Desktop\\pr_project_1\\trainingSymbols\\"

    print(datetime.datetime.now())

    UIToFeaturesDict, UIToLabelDict = {}, {}
    size, file_counter = 0, 0

    start_read_time = time.time()
    for file in os.listdir(file_path):
        file_counter += 1
        if file.endswith(".inkml"):
            # add a parameter visualize and set it to true to see features
            if preprocess:
                UI, features = extractFeatures_with_preprocessing.getFeatures(file_path + file, DPI)
            else:
                UI, features = extractFeatures.getFeatures(file_path + file, DPI)
            if size == 0:
                size = features.shape[0]
            UIToFeaturesDict[UI] = features
            if file_counter % 100 == 0:
                print(file_counter)
    end_feature_time = time.time()
    print(f"Time taken for features: {end_feature_time - start_read_time}")
    # """
    with open(file_path + "iso_GT.txt") as reader:
        for line in reader:
            tempList = line.split(",")
            tempList[1] = tempList[1].replace("\\", "")
            UIToLabelDict[tempList[0]] = tempList[1].replace('\n', '')
    # """
    read_time = time.time()
    print(f"Time to read feature file: {read_time - end_feature_time}")
    featureArr, UIArr, labelArr = [], [], []
    i = 0
    for key in UIToFeaturesDict:
        i += 1
        featureArr.append([UIToFeaturesDict[key]])
        labelArr.append(UIToLabelDict[key])
        UIArr.append(key)
        if i % 1000 == 0:
            print(i)
    np_time = time.time()
    print(f"NP array time taken: {np_time - read_time}")
    featureFile = open(f"featureFile_withPreprocessing_{DPI}dpi_f{f}_TRAINING", "wb")
    pickle.dump([np.asarray(UIArr), np.asarray(featureArr), np.asarray(labelArr)], featureFile)
    featureFile.close()
    final_time = time.time()
    print(f"Entire process took: {final_time - start_read_time}")
    print(datetime.datetime.now())


if __name__ == '__main__':
    main()
