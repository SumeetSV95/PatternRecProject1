import pickle
import numpy as np


def getMaxProb(classifier, samples, n):
    probabilities = classifier.predict_proba(samples)
    classes = classifier.classes_
    selected_classes = set()
    selected_classes_list = []
    top_classes = []
    for k in range(len(probabilities)):
        selected_classes = set()
        selected_classes_list = []
        for i in range(n):
            max_prob = float('-inf')
            index = -1
            for j in range(len(probabilities[k])):
                if probabilities[k][j] > max_prob and classes[j] not in selected_classes:
                    max_prob = probabilities[k][j]
                    index = j
            if (len(classes[index]) > 1 and classes[index] != 'COMMA') or classes[index] == '{' or classes[
                index] == '}':
                selected_classes_list.append("\\" + classes[index])
            else:
                selected_classes_list.append(classes[index])
            selected_classes.add(classes[index])
        top_classes.append(selected_classes_list)

    return top_classes


def find_labels(preprocess, isSVM):
    # ---------------------------------------------------------------------------------------------------
    test_feature_file_name = "features_TESTING"
    if preprocess:
        test_feature_file_name += "_preprocessing"

    with open(test_feature_file_name, 'rb') as fp:
        data = pickle.load(fp)
    # ---------------------------------------------------------------------------------------------------
    model_file = ""
    if isSVM:
        model_file += "SVM"
    else:
        model_file += "RFC"
    if preprocess:
        model_file += "_preprocessing"

    with open(model_file, 'rb') as fp:
        classifier = pickle.load(fp)
    # ---------------------------------------------------------------------------------------------------
    features = data[1]
    filenames = data[0]
    # ---------------------------------------------------------------------------------------------------
    all_features = []
    for i in range(len(features)):
        all_features.append(features[i][0])
    features = np.asarray(all_features)
    # ---------------------------------------------------------------------------------------------------
    with open(model_file + ".txt", 'w') as fp:
        predicted_classes = getMaxProb(classifier, features, 10)
        for i in range(len(features)):
            fp.write(filenames[i] + ",")
            fp.write(",".join(predicted_classes[i]) + '\n')
    print(f"Created the result text file: ({model_file+'.txt'})")
