""" Preprocess Input """

# imports
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import os


# divide datasets into stratified train and test
def divideDataSets(basePath, outputPath):
    # get file paths in base path
    onlyfiles = [(join(basePath, f), f.replace('.csv', ''))
                 for f in listdir(basePath) if isfile(join(basePath, f))]
    # divide
    finalPaths = []
    for file in onlyfiles:
        # read csv
        dataset = pd.read_csv(file[0], engine='c')
        # target
        target = dataset.iloc[:, 0]
        # features
        features = dataset.iloc[:, 1: len(dataset.columns)-1]
        # train / test
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, stratify=target, train_size=0.67)
        # join target
        train = X_train.copy()
        train.insert(0, y_train.name, y_train.values)
        test = X_test.copy()
        test.insert(0, y_test.name, y_test.values)
        # base path file
        basePath = outputPath+f'\_{file[1]}'
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        # print to file
        pathTrainTest = (
            f'{basePath}\\train.csv', f'{basePath}\\test.csv')
        train.to_csv(pathTrainTest[0], index=False)
        test.to_csv(pathTrainTest[1], index=False)
        # print to file visualize
        pathVisualize = (
            f'{basePath}\\train_visualize.txt', f'{basePath}\\test_visualize.txt')
        with open(pathVisualize[0], 'w') as trainFile, open(pathVisualize[1], 'w') as testFile:
            trainFile.write(str(train))
            testFile.write(str(test))
        # add paths
        finalPaths.append(pathTrainTest)
    # return
    return finalPaths


"""
Useful base resources
https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/
"""
