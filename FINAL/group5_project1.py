"""
Group 5:
    Abhay Kulkarni (ak6277), Sumeet Vadhavkar (sv6234), Yash Wagle (yw5262)

Pattern Recognition:
    Project 1

Needs:
    saveFeatures.py, extractFeatures_with_preprocessing.py, extractFeatures.py
    two_classifiers.py
    generate_results.py

I/P parameters:
    group5_project1.py <training_file> <test_file>
    TODO: should we let him given i/p for preprocessing and classifier

"""
import saveFeatures
import two_classifiers
import generate_results


def main():
    # ---------------------------------------------------------------------------------------------------
    # Pass these as code parameters
    training_file = "C:\\Users\\Abhay\\Desktop\\pr_project_1\\trainingSymbols\\"  # inkml files
    test_file = "C:\\Users\\Abhay\\Desktop\\pr_project_1\\testSymbols\\"  # inkml files
    is_preprocessing = True  # bool
    is_SVM = True  # False --> Random Forest
    # ---------------------------------------------------------------------------------------------------
    # Generate feature file for TEST and TRAIN (pickle dump)
    is_training = True
    for file in [training_file, test_file]:
        saveFeatures.save_features(file_path=file, preprocess=is_preprocessing, isTrain=is_training)
        is_training = False
    # ---------------------------------------------------------------------------------------------------
    # Train model using generated TRAINING feature file (pickle dump)
    two_classifiers.model(isSVM=is_SVM, preprocess=is_preprocessing)
    # ---------------------------------------------------------------------------------------------------
    # Generate classification result text file for TEST set
    generate_results.find_labels(preprocess=is_preprocessing, isSVM=is_SVM)
    # ---------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
