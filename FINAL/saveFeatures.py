import os
import pickle
import time
import numpy as np
import datetime
import extractFeatures
import extractFeatures_with_preprocessing

DPI = 50  # Found these parameter to be best when considering time taken to train
isPrint = True  # Print progress


def parse_inkml_to_feature(filename, is_preprocess):
    """
    Parses the inkml files to feature vectors
    :param filename: Input filename
    :param is_preprocess: preprocessing flag
    :return: Feature vector
    """
    UIToFeaturesDict, UIToLabelDict = {}, {}
    size, file_counter = 0, 0
    for file in os.listdir(filename):
        file_counter += 1
        if file.endswith(".inkml"):
            if is_preprocess:
                # Pre-processing "smooths" the hooks and sharp corners
                UI, features = extractFeatures_with_preprocessing.getFeatures(filename + file, DPI)
            else:
                UI, features = extractFeatures.getFeatures(filename + file, DPI)
            if size == 0:
                size = features.shape[0]
            UIToFeaturesDict[UI] = features
            if isPrint:
                if file_counter % 1000 == 0:
                    print(file_counter)
    return UIToFeaturesDict, UIToLabelDict


def update_label_for_train(filename, input_dict):
    """
    Adds labels to feature vectors for training set
    :param filename: input file
    :param input_dict: feature dict
    :return: updates feature dict
    """
    with open(filename + "iso_GT.txt") as reader:
        for line in reader:
            tempList = line.split(",")
            tempList[1] = tempList[1].replace("\\", "")
            input_dict[tempList[0]] = tempList[1].replace('\n', '')
    return input_dict


def create_pickle_dump(feature_map, label_map, isTrain, is_preprocessing):
    """
    Creates the feature dump using feature vector along with the labels
    """
    featureArr, UIArr = [], []
    labelArr = []
    i = 0
    for key in feature_map:
        i += 1
        featureArr.append([feature_map[key]])
        if isTrain:
            labelArr.append(label_map[key])
        UIArr.append(key)
        if isPrint:
            if i % 1000 == 0:
                print(i)
    # ---------------------------------------------------------------------------------------------------
    type_of_data = "TRAINING" if isTrain else "TESTING"
    data_array = [np.asarray(UIArr), np.asarray(featureArr), np.asarray(labelArr)]
    final_file_name = f"features_{type_of_data}"
    if is_preprocessing:
        final_file_name += "_preprocessing"
    featureFile = open(final_file_name, "wb")
    pickle.dump(data_array, featureFile)
    featureFile.close()


def save_features(file_path, preprocess, isTrain):
    print(datetime.datetime.now())
    # ---------------------------------------------------------------------------------------------------
    start_read_time = time.time()
    UIToFeaturesDict, UIToLabelDict = parse_inkml_to_feature(filename=file_path, is_preprocess=preprocess)

    end_feature_time = time.time()
    print(f"Time taken for features: {end_feature_time - start_read_time}")
    # ---------------------------------------------------------------------------------------------------
    if isTrain:
        UIToLabelDict = update_label_for_train(filename=file_path, input_dict=UIToLabelDict)
    # ---------------------------------------------------------------------------------------------------
    read_time = time.time()
    print(f"Time to read feature file: {read_time - end_feature_time}")
    create_pickle_dump(feature_map=UIToFeaturesDict, label_map=UIToLabelDict, isTrain=isTrain,
                       is_preprocessing=preprocess)
    final_time = time.time()
    print(f"Feature saving took: {final_time - start_read_time}")
    print(datetime.datetime.now())
    # ---------------------------------------------------------------------------------------------------
