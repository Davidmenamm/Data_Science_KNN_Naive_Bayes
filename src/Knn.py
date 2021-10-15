""" Run KNN algorithm for binary classification """

# imports
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from sklearn import metrics


# Knn
def MultipleKnn(trainTestPaths):
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
        calcPath = str(pPath)+'\calculations'
        if not os.path.exists(calcPath):
            os.makedirs(calcPath)
        # Knn with multiple distances: 1 is manhattan and 2 is euclidian
        for dist in range(1, 2+1):
            # distance dir
            distName = 'manhattan' if dist == 1 else 'euclidian'
            distancePath = calcPath + f'\distance_{distName}'
            if not os.path.exists(distancePath):
                os.makedirs(distancePath)
            # Odd k values
            kvals = []
            accurracies = []
            for k in range(1, 20):  # featTrain.shape[1]):
                # odd numbers from 1 to 30
                if k % 2 != 0:
                    # run classification
                    knn = KNeighborsClassifier(n_neighbors=k, p=dist)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    distances = knn.kneighbors(
                        X_test, return_distance=True)
                    # metrics
                    acc = metrics.accuracy_score(y_test, y_pred)
                    # append ks and metrics
                    kvals.append(k)
                    accurracies.append(acc)
                    # print results in files
                    classifPath = distancePath + f'\classif_k{k}.csv'
                    strPred = str(y_pred)
                    strReal = str(y_test.values)
                    # print(type(distances[1].tolist()), distances[1].tolist())
                    distancePoints = []
                    for distPoint in zip(distances[1].tolist(), distances[0].tolist()):
                        distancePoints.append(
                            list(zip(distPoint[0], distPoint[1])))
                    strDistances = ''
                    for point in distancePoints:
                        strDistances = strDistances + \
                            (str(point) + '\n')
                    with open(classifPath, 'w') as predFile:
                        predFile.write(f'K values is: {k}\n\n\n')
                        predFile.write(f'Accurracy is: {acc}\n\n')
                        predFile.write(
                            f'Predicted Target:\n{strPred}\n\n')
                        predFile.write(
                            f'Real Target:\n{strReal}\n\n\n')
                        predFile.write(f'Distances:\n{strDistances}')
            # matplot print
            kValuesNp = np.asarray(kvals, dtype=np.int32)
            acurracyNp = np.asarray(accurracies, dtype=np.float32)
            plt.title(f'{baseName} - {distName}')
            plt.plot(kValuesNp, acurracyNp)
            plt.xlabel('Value of K for KNN')
            plt.ylabel('Testing Accuracy')
            plt.show()
