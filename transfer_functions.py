

def identity(x, derivate=False):
    return x if not derivate else np.ones(x.shape)


def logistic(x, derivate=False):
    return 1 / (1 + np.e ** (-x)) if not derivate else x * (1 - x)


def hyperbolic_tangent(x, derivate=False):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1) if not derivate else 1 - x ** 2


def tanh(x, derivate=False):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) if not derivate else 1 - np.power(tanh(x), 2)

def relu(x, derivate=False):
    # TODO 2
    return_val = []

    if not derivate:
        for y in x:
            return_val.append(max(y, 0))
    else:
        for y in x:
            if y > 0:
                return_val.append(1)
            else:
                return_val.append(0)

    return np.array(return_val)
