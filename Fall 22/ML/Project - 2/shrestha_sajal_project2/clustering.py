import numpy as np


def randomSample(arr: np.array, n: int = 1):
    return arr[np.random.choice(len(arr), size=n, replace=False)]


def dist(X, Y):
    return np.linalg.norm(X - Y)


def K_Means(X, K, mu):
    X = np.array(X)
    mu = np.array(mu)

    if type(X[0]) == np.ndarray:
        noOfFeatures = len(X[0])
    else:
        noOfFeatures = 1
    if len(mu) == 0:
        mu = randomSample(X, K)
    iteration = 0
    while 1:
        distancesOfXsToMus = [[dist(eachMu, eachXTrain)
                               for eachMu in mu] for eachXTrain in X]
        clusterWithRespectToPoint = [
            np.argmin(distanceWithMu) for distanceWithMu in distancesOfXsToMus]
        newMu = np.empty([0, noOfFeatures])
        iteration += 1
        for muPosition in np.arange(0, len(mu)):
            if noOfFeatures == 1:
                newMu = np.append(newMu, np.mean(
                    X[np.where(clusterWithRespectToPoint == muPosition)]))
            else:
                newMu = np.append(newMu, [np.mean(
                    X[np.where(clusterWithRespectToPoint == muPosition)], axis=0)], axis=0)
        for i in np.argwhere(np.isnan(newMu)):
            newMu[i] = mu[i]
        if np.array_equal(mu, newMu):
            return mu
        else:
            mu = newMu
            continue


def K_Means_better(X, K):
    mus = [K_Means(X, K, [])]
    for i in range(0, len(X)):
        mus = np.vstack((mus, [K_Means(X, K, [])]))
    values, counts = np.unique(mus, axis=0, return_counts=True)
    maximumEpochs = len(X)*100
    counter = 0
    while counter < maximumEpochs:
        values, counts = np.unique(mus, return_counts=True, axis=0)
        if max(counts) >= 0.4*len(mus):
            return values[np.argmax(counts)]
        mus = np.vstack((mus, [K_Means(X, K, [])]))
        counter += 1

    values, counts = np.unique(mus, return_counts=True, axis=0)
    return values[np.argmax(counts)]
