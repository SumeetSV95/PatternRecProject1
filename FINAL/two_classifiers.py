import datetime
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def do_random_forest(features, labels):
    num_of_trees, maximum_depth = 90, 35
    clf = RandomForestClassifier(n_estimators=num_of_trees, criterion="entropy",
                                 max_depth=maximum_depth, max_features="sqrt",
                                 n_jobs=-1)
    clf.fit(features, labels)
    return clf


def do_svm():
    return 0


def model(isSVM, preprocess):
    filename = f"features_TRAINING"
    if preprocess:
        filename += "_preprocessing"

    code_start = time.time()
    print(datetime.datetime.now())

    with open(filename, 'rb') as fp:
        objects = pickle.load(fp)
    features = objects[1]
    labels = objects[2]
    total_samples = len(features)
    train_features = []
    f_time = time.time()
    print(f"Read features in :{f_time - code_start} s")

    for i in range(len(features)):
        train_features.append(features[i][0])
    features = np.asarray(train_features)

    if not isSVM:
        classifier = do_random_forest(features, labels)
    else:
        classifier = do_svm()

    """
    num_samples = int(total_samples / nt)
    # Default bootstrap=True
    model = SVC(kernel='rbf', probability=True, class_weight='balanced', verbose=1)
    
    clf = OneVsRestClassifier(BaggingClassifier(model, bootstrap=False, n_estimators=nt, n_jobs=-1, verbose=1),
                              n_jobs=-1)
    
    """

    classifier_file = ""
    if isSVM:
        classifier_file += "SVM_"
    else:
        classifier_file += "RFC_"
    if preprocess:
        classifier_file += "preprocessing"

    featureFile = open(classifier_file, "wb")
    pickle.dump(classifier, featureFile)
    featureFile.close()
    print(f"Model fit in: {time.time() - code_start}")
