""" Manage All the Program """

# imports
from Preprocessing import divideDataSets
from Knn import MultipleKnn

# input paths
pathInHypothesis = r'data\input'

# output paths
mainOutputPath = r'data\output'
# path


# Coordinator
def coordinate():
    # divide datasets into train and test
    trainTestPaths = divideDataSets(pathInHypothesis, mainOutputPath)
    # knn for multiple k and distance, for multiple hypothesis
    MultipleKnn(trainTestPaths)
