import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def main():
    with open('/Users/sumeet95/Downloads/featureFile', 'rb') as fp:
        _, features, labels = pickle.load(fp)

    sampleList = []
    trainingFeatures = []
    for i in range(features.shape[0]):
        trainingFeatures.append(features[i][0])
        sampleList.append(i)
    trainingFeatures = np.asarray(trainingFeatures)
    sampleList = np.asarray(sampleList)
    rgn = np.random.default_rng()
    randomSamples = rgn.choice(sampleList, 30000, replace=False)
    trainingFeatures = trainingFeatures[randomSamples, :]
    trainingFeatures, testingFeatures = np.split(trainingFeatures, [20000])
    labels = labels[randomSamples]
    trainingLabels, testLabels = np.split(labels, [20000])
    print(trainingFeatures.shape, trainingLabels.shape)

    trainingLabelSet = set()
    for i in trainingLabels:
        trainingLabelSet.add(i)
    testSamples = []
    for i, label in enumerate(testLabels):
        if label in trainingLabelSet:
            testSamples.append(i)
    testingFeatures = testingFeatures[testSamples, :]
    testLabels = testLabels[testSamples]
    print(testingFeatures.shape, testLabels.shape)
    maxAccuracy = 0

    treeSizes = [100, 150, 200, 250]
    maxDepthSizes = [4, 6, 8, 10, 12, 14, 16]
    minSampleLeaves = [8, 12, 16, 20]
    minSamplesSplit = [15, 30, 45, 60]
    for treeSizes in treeSizes:
        for maxDepth in maxDepthSizes:
            for minSampleLeaf in minSampleLeaves:
                for m in minSamplesSplit:
                    clf = RandomForestClassifier(n_estimators=treeSizes, criterion='entropy', max_depth=maxDepth,
                                                 min_samples_split=m, min_samples_leaf=minSampleLeaf, n_jobs=-1)
                    clf.fit(trainingFeatures, trainingLabels)
                    score = clf.score(testingFeatures, testLabels)
                    if score > maxAccuracy:
                        fileName = "RandomForestClassifiers/TreeSize_{}_maxDepth_{}_minSamplesLeaf_{}_minSamplesSplit_{}".format(
                            treeSizes,
                            maxDepth,
                            minSampleLeaf, m)
                        bestParams = [treeSizes, maxDepth, minSampleLeaf, m]
                        maxAccuracy = score
                        print(score)
                        print(bestParams)
                        featureFile = open(fileName, "wb")
                        pickle.dump(clf, featureFile)
                        featureFile.close()

            print(treeSizes, maxDepth)


"""    for i in range(len(features)):
        train_features.append(features[i][0])
    features = np.asarray(train_features)

    clf = svm.SVC(kernel='rbf', probability=True)
    print("Learning started")
    clf.fit(features, labels)
    print("Learning complete")
    correct = 0
    wrong = 0
    featureFile = open("/content/SVM_Model_v2", "wb")
    pickle.dump(clf, featureFile)
    featureFile.close()"""

if __name__ == '__main__':
    main()
