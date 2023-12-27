import numpy as np
import math
from typing import List
from statistics import mode

# 2. Decision Trees

def entropy(probabilities: List[float]):
    entropy = 0
    for probabilityIndex in probabilities:
        if probabilityIndex > 0:
            entropy -= probabilityIndex * math.log2(probabilityIndex)
    return entropy


def informationGain(totalY: List[int], yWhenXisYes: List[int], yWhenXisNo: List[int]):

    totalY = np.array(totalY)
    yWhenXisYes = np.array(yWhenXisYes)
    yWhenXisNo = np.array(yWhenXisNo)

    nTotal1 = np.count_nonzero(totalY == 1)
    nTotal0 = np.count_nonzero(totalY == 0)
    nTotal = len(totalY)
    nXEqualsYes1 = np.count_nonzero(yWhenXisYes == 1)
    nXEqualsYes0 = np.count_nonzero(yWhenXisYes == 0)
    nXEqualsYes = len(yWhenXisYes)
    nXEqualsNo1 = np.count_nonzero(yWhenXisNo == 1)
    nXEqualsNo0 = np.count_nonzero(yWhenXisNo == 0)
    nXEqualsNo = len(yWhenXisNo)

    if (nTotal == 0 or nTotal1 == 0 or nTotal0 == 0):
        return 0

    entropyTotal = entropy([nTotal1/nTotal, nTotal0/nTotal])
    entropyXEqualsYes = 0 if (nXEqualsYes1 == 0 or nXEqualsYes0 == 0) else entropy(
        [nXEqualsYes1/nXEqualsYes, nXEqualsYes0/nXEqualsYes])
    entropyXEqualsNo = 0 if (nXEqualsNo1 == 0 or nXEqualsNo0 == 0) else entropy(
        [nXEqualsNo1/nXEqualsNo, nXEqualsNo0/nXEqualsNo])

    informationGain = entropyTotal - \
        (nXEqualsYes/nTotal*entropyXEqualsYes) - \
        (nXEqualsNo/nTotal*entropyXEqualsNo)

    return informationGain


def findIG(X, Y):
    xEqualsYes = np.array(Y[X == 1]).flatten()
    xEqualsNo = np.array(Y[X == 0]).flatten()
    return informationGain(Y, xEqualsYes, xEqualsNo)


def igOfAllProvided(X, Y):
    IGOfAllGiven = []
    for x in X:
        ig = findIG(x, Y)
        IGOfAllGiven.append(ig)
    return IGOfAllGiven

################################## Binary values ##################################


def maximumFreqNp(npArray):
    values, counts = np.unique(npArray, return_counts=True)
    return (values[np.argmax(counts)])


def conditionToBranch(X, Y, remainingX, features, maxDepth):
    if maxDepth <= features-len(remainingX) or len(remainingX) <= 0:
        return False
    ig = igOfAllProvided(X, Y)
    if max(ig) == 0:
        return False
    return ig


def binaryBranching(X, Y, remainingX):
    ig = igOfAllProvided(X, Y)
    branchIndex = np.argmax(ig)
    branchFeature = remainingX[branchIndex]
    remainingX = remainingX[remainingX != branchFeature]

    y1 = Y[X[branchIndex] > 0]
    y0 = Y[X[branchIndex] <= 0]
    x0 = []
    x1 = []
    for numberOfX in range(len(X)):
        if numberOfX != branchIndex:
            x0.append(X[numberOfX][X[branchIndex] <= 0])
            x1.append(X[numberOfX][X[branchIndex] > 0])
    return [{"x": x0, "y": y0, "remainingX": np.array(remainingX).flatten()},
            {"x": x1, "y": y1, "remainingX": np.array(remainingX).flatten()}]


def createBranch(X, Y, remainingX, features, maxDepth):
    condition = conditionToBranch(
        X, Y, remainingX, features, maxDepth)
    if condition:
        indexOfMaxIg = np.argmax(condition)
        featureToBranchOn = remainingX[indexOfMaxIg]
        branchingStep = binaryBranching(X, Y, remainingX)
        return {"positionOfX": featureToBranchOn,
                "divide": [createBranch(branchingStep[0]["x"], branchingStep[0]["y"], branchingStep[0]["remainingX"], features, maxDepth),
                           createBranch(branchingStep[1]["x"], branchingStep[1]["y"], branchingStep[1]["remainingX"], features, maxDepth)]}
    else:
        return {"positionOfX": None,
                "divide": maximumFreqNp(Y)}


def makePrediction(x, DT):
    positionOfX = DT["positionOfX"]
    divide = DT["divide"]
    while positionOfX != None:
        if x[positionOfX] <= 0:
            positionOfX = divide[0]["positionOfX"]
            divide = divide[0]["divide"]
        else:
            positionOfX = divide[1]["positionOfX"]
            divide = divide[1]["divide"]
    return divide


def DT_train_binary(X, Y, max_depth):
    features = len(X[0])
    maxDepth = min(features, features if (
        max_depth == -1) else max_depth)
    remainingX = np.arange(features)
    X = np.array(X).T
    Y = np.array(Y)
    DT = createBranch(X, Y, remainingX, features, maxDepth)
    return DT


def DT_test_binary(X, Y, DT):
    yPrediction = np.array(
        [makePrediction(xInstance, DT) for xInstance in X])
    correctPredictions = np.sum(Y == yPrediction)
    totalTestDatas = len(Y)
    accuracy = (correctPredictions/totalTestDatas)
    return accuracy


################################## Real values ##################################


def findY(x, Y, option):
    x = np.array(x)
    Y = np.array(Y)

    below = Y[x <= option]
    above = Y[x > option]
    return [above, below]


def findIgGivenXOptions(x, Y, option):
    x0 = np.array(x)
    Y = np.array(Y)

    aboves, belows = findY(x0, Y, option)
    ig = informationGain(Y, aboves, belows)
    return ig


def findBranchingOptions(x):
    x = np.array(x)
    return np.unique(x)


def IGWithGivenX(x, Y):
    branchingOptions = findBranchingOptions(x)
    igs = [findIgGivenXOptions(x, Y, option)
           for option in branchingOptions]
    branchingValueForMaxIG = branchingOptions[np.argmax(igs)]
    maxIG = np.max(igs)
    return [branchingValueForMaxIG, maxIG]


def dataForMaxIG(X, Y):
    possibleBranches = np.array([IGWithGivenX(x, Y) for x in X])
    maxIG = np.max(possibleBranches.T[1])
    postitionOfX = np.argmax(possibleBranches.T[1])
    branchOfX = possibleBranches.T[0][postitionOfX]
    return (postitionOfX, branchOfX, maxIG)


def conditionToBranchRealValues(X, Y, maxDepth, depth):
    if maxDepth <= depth:
        return False
    positionOfX, branchOfX, maxIG = dataForMaxIG(X, Y)
    if maxIG == 0:
        return False
    else:
        return ([positionOfX, branchOfX])


def getBranchingSteps(X, Y, positionOfX, branchOfX):
    xForXGreater = [x[X[positionOfX] > branchOfX] for x in X]
    yForXGreater = Y[X[positionOfX] > branchOfX]
    xForXLessOrEqual = [x[X[positionOfX] <= branchOfX] for x in X]
    yForKLessOrEqual = Y[X[positionOfX] <= branchOfX]

    return [{"x": xForXLessOrEqual,
             "y": yForKLessOrEqual},
            {"x": xForXGreater,
             "y": yForXGreater}]


def createBranchForRealValues(X, Y, max_depth, depth):
    condition = conditionToBranchRealValues(X, Y, max_depth, depth)
    if condition:
        positionOfX = condition[0]
        branchOfX = condition[1]

        branching_step = getBranchingSteps(
            X, Y, positionOfX, branchOfX)

        return {"positionOfX": positionOfX,
                "divideValue": branchOfX,
                "divide": [createBranchForRealValues(branching_step[0]["x"],
                                           branching_step[0]["y"],
                                           max_depth,
                                           depth+1),
                        createBranchForRealValues(branching_step[1]["x"],
                                           branching_step[1]["y"],
                                           max_depth,
                                           depth+1)]}
    else:
        return {"positionOfX": None,
                "divideValue": None,
                "divide": maximumFreqNp(Y)}


def allPossibleBranchings(transposeOfInput):
    allPossibleBranching = np.array([findBranchingOptions(x)
                               for x in transposeOfInput], dtype='object')
    return allPossibleBranching


def totalPossibleDepth(transposeOfInput):
    allPossibleBranching = allPossibleBranchings(transposeOfInput)
    totalPossibleDepth = np.sum(np.array([len(x) for x in allPossibleBranching]))
    return totalPossibleDepth

def makePredictionForRealValues(x, DT):
    positionOfX = DT["positionOfX"]
    divide = DT["divide"]
    divideValue = DT["divideValue"]
    while positionOfX != None:
        if x[positionOfX] <= divideValue:
            positionOfX = divide[0]["positionOfX"]
            divideValue = divide[0]["divideValue"]
            divide = divide[0]["divide"]
        else:
            positionOfX = divide[1]["positionOfX"]
            divideValue = divide[1]["divideValue"]
            divide = divide[1]["divide"]
    return divide

def DT_train_real(X, Y, max_depth):
    X = np.array(X).T
    Y = np.array(Y)

    totalPossibleDepth = totalPossibleDepth(X)
    max_depth = min(totalPossibleDepth, totalPossibleDepth if (
        max_depth == -1) else max_depth)

    depth = 0

    DT = createBranchForRealValues(X, Y, max_depth, depth)
    return DT

def DT_test_real(X, Y, DT):
    X = np.array(X)
    Y = np.array(Y)
    yPrediction = [makePredictionForRealValues(x, DT) for x in X]
    correctPredictions = np.sum(Y == yPrediction)
    totalTestDataCount = len(Y)
    accuracy = (correctPredictions/totalTestDataCount)*100
    return accuracy


########################## Question no. 3: Random forest ##########################

def RF_build_random_forest(samples, labels, max_depth, tree_count):
    forest = []
    for tree in range(tree_count):
        i = np.random.randint(0, len(samples), int(0.1 * len(samples)))
        trainX = [samples[s] for s in i]
        trainY = [labels[s] for s in i]
        trees = DT_train_binary(trainX, trainY, max_depth)

        forest.append(trees)

    return forest


def RF_test_one(x, forest):
    predictions = []
    predictions = [makePrediction(x, tree) for tree in forest]
    predictions.append(mode(predictions))

    return predictions


def RF_test_random_forest(samples, labels, forest):
    totalPredictions = [RF_test_one(x, forest) for x in samples]
    predicitionTranspose = np.array(totalPredictions).T
    sumOfCorrespondingPredicition = [sum(
        labels == eachPredicitionTranspose) for eachPredicitionTranspose in predicitionTranspose]
    accuracyList = [sm/len(labels) for sm in sumOfCorrespondingPredicition]

    for i in range(len(accuracyList)-1):
        print("DT",i,":  ", "%.5f" % accuracyList[i])

    return accuracyList[-1]
