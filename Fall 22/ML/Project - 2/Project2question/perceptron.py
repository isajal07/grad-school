
def perceptron_train(X, Y):
    dims = len(X[0])
    weights = []
    for i in range(dims):
        weights.append(0)
    bias = 2
    update_count = 1
    while update_count > 0:
        update_count = 0
        for i in range(len(X)):
            result = bias
            for dim in range(dims):
                result += X[i][dim] * weights[dim]
            if not ((result <= 0 and Y[i] <= 0) or (result > 0 and Y[i] > 0)):
                update_count += 1
                for dim in range(dims):
                    weights[dim] = weights[dim] + Y[i] * X[i][dim]
                    bias = bias + Y[i]
    return weights, bias

def perceptron_test(X_test, Y_test, w, b : float):
    accuracy = 0
    dims = len(X_test[0])
    for i in range(len(X_test)):
        result = b
        for dim in range(dims):
            result += X_test[i][dim] * w[dim]
        if not result == 0:
            if result > 0 and Y_test[i] > 0:
                accuracy += 1
            elif result < 0 and Y_test[i] < 0:
                accuracy += 1
        else:
            accuracy += 1
    accuracy = accuracy / len(X_test)
    return accuracy