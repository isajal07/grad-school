import numpy as np

def build_nparray(data):
    arrayData = np.array(data)
    arrayData1 = np.array(arrayData[1:])
    featureValues = np.array([x[0:len(x) - 1] for x in arrayData1]).astype(float)
    labels = np.array([x[len(x) - 1] for x in arrayData1]).astype(int)
    
    return featureValues, labels


def build_list(data):
    myData = data[1:].tolist()
    labels = [x[len(x) - 1] for x in myData]
    
    intLabels = []
    for item in labels:
        intLabels.append(int(item))

    featureValues = [x[0:len(x) - 1] for x in myData]
    featuresData = []
    for item in featureValues:
        features = []
        for x in item:
            features.append(float(x))
        featuresData.append(features)
    
    return featuresData, intLabels


def build_dict(data):
    myData = data.tolist()

    listOfDict = []
    keys = myData[0][:-1]
    for item in range(1, len(myData)):
        featureDict = {}
        countA = 0
        for key in keys:
            featureDict[key] = float(myData[item][countA])
            countA += 1
        listOfDict.append(featureDict)

    labelsDict = {}
    countB = 0
    for item in range(1, len(myData)):
        labelsDict[countB] = int(myData[item][-1])
        countB += 1
    
    return listOfDict, labelsDict


