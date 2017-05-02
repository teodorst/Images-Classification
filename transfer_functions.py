import numpy as np

def identity(x, derivate=False):
    return x if not derivate else np.ones(x.shape)


def logistic(x, derivate=False):
    return 1 / (1 + np.exp(-x)) if not derivate else np.multiply(x, (1 - x))


def hyperbolic_tangent(x, derivate=False):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1) if not derivate else 1 - x ** 2


def tanh(x, derivate=False):
    return np.tanh(x) if not derivate else 1 - x ** 2

def relu_def(x):
    return max(x, 0)

def relu_deriv(x):
    return 1 if x > 1 else 0

def relu(x, derivate=False):
    if derivate:
        return np.vectorize(relu_deriv, otypes=[np.float])(x)
    return np.vectorize(relu_def, otypes=[np.float])(x)


# def relu(x, derivate=False):
#     # TODO 2
#     return_val = []
#     if not derivate:
#         for y in x:
#             return_val.append(max(y, 0))
#     else:
#         for y in x:
#             if y > 0:
#                 return_val.append(1)
#             else:
#                 return_val.append(0)

#     return np.array(return_val)
