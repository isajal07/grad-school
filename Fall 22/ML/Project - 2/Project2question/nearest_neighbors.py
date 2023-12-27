import numpy as np


def maxFrequency(ysOfSmallestDistances):
    values, counts = np.unique(ysOfSmallestDistances, return_counts=True)
    
    return (values[np.argmax(counts)])


def predict_KNN(xTrain, yTrain, xTestPoints, K):
    distancesWithTrainingX = [np.linalg.norm(xTestPoints - trainXs) for trainXs in xTrain]
    
    indexOfSmallestDistance = np.argpartition(distancesWithTrainingX, K)[:K]
    ysOfSmallestDistances = [yTrain[i] for i in indexOfSmallestDistance]
    
    predictedYs = maxFrequency(ysOfSmallestDistances)
    
    return predictedYs


def find_accuracy(yTest, predictedYs):
    predictedYs = np.array(predictedYs)
    noOfCorrectPrediction = np.sum(yTest == predictedYs)
    totalTestDataCount = len(yTest)
    accuracy = (noOfCorrectPrediction/totalTestDataCount)
    
    return accuracy


def KNN_test(X_train, Y_train, X_test, Y_test, K):
    xTrain = np.array(X_train)
    yTrain = np.array(Y_train)
    xTest = np.array(X_test)
    yTest = np.array(Y_test)

    predictedYs = [predict_KNN(xTrain, yTrain, xTestPoints, K) for xTestPoints in xTest]

    accuracy = find_accuracy(yTest, predictedYs)
    
    return accuracy

def choose_K(X_train,Y_train,X_val,Y_val):
    accuraciesInKs=np.array([KNN_test(X_train,Y_train,X_val,Y_val,K) for K in range(1,len(Y_train))])
    
    bestK = np.argmax(np.array([accuraciesInKs]))+1
    
    return bestK