""" Run Naive Bayes for multiple hypothesis """

# imports
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from pathlib import Path
from sklearn import metrics


# multiple naive bayes
def MultipleNB(trainTestPaths):
    # parent directories (folder)
    parentPaths = []
    for trainTestPath in trainTestPaths:
        trainPath = Path(trainTestPath[0])
        parentPaths.append(trainPath.parent.absolute())
        # iterate calculations through all hypothesis
    for pPath, trainTestPath in zip(parentPaths, trainTestPaths):
                            # get base folder name
        baseName = os.path.basename(os.path.normpath(pPath))
        # train and test dataset
        train = pd.read_csv(trainTestPath[0])
        test = pd.read_csv(trainTestPath[1])
        # get target
        y_train = train.iloc[:, 0]
        y_test = test.iloc[:, 0]
        # get features
        X_train = train.iloc[:, 1: len(train.columns)-1]
        X_test = test.iloc[:, 1: len(test.columns)-1]
        # new dir if not created
        calcPath = str(pPath)+'\calculations_NB'
        if not os.path.exists(calcPath):
            os.makedirs(calcPath)
        # run classification
        nbc = GaussianNB()
        nbc.fit(X_train, y_train)
        y_pred = nbc.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        # print results in files
        classifPath = calcPath + f'\classif_{baseName}.csv'
        strPred = str(y_pred)
        strReal = str(y_test.values)
        with open(classifPath, 'w') as predFile:
            predFile.write(f'Naive Bayes for: {baseName}\n\n\n')
            predFile.write(f'Accurracy is: {acc}\n\n')
            predFile.write(
                f'Predicted Target:\n{strPred}\n\n')
            predFile.write(
                f'Real Target:\n{strReal}\n\n\n')
