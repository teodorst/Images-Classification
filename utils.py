import numpy as np

from numpy.linalg import norm

def close_enough(arr1, arr2, max_err = 0.0000001):
    assert(arr1.shape == arr2.shape)
    return norm(arr1.reshape(arr1.size) - arr2.reshape(arr2.size)) < max_err


def im2col_indices(A, size, stepsize=1):
    # Parameters
    D,M,N = A.shape
    col_extent = N - size[1] + 1
    row_extent = M - size[0] + 1

    # Get Starting block indices
    offset_idx = np.arange(size[0])[:,None] * N + np.arange(size[1])

    # Generate Depth indeces
    didx = M * N * np.arange(D)

    offset_idx=(didx[:,None] + offset_idx.ravel()).reshape((-1, size[0], size[1]))

    # Get offsetted indices, absolute to first element 0,
    # across the height and width of input array
    start_idx = np.arange(row_extent)[:,None] * N + np.arange(col_extent)
    indices = (start_idx.ravel()[:,None] + offset_idx.ravel())

    if stepsize > 1:
        # Remove unused squares
        stepsize_idx = []
        for i in range(indices.shape[0]):
            block = indices[i][0] // M
            if block % stepsize == 0 and i % stepsize == 0:
                stepsize_idx.append(i)
        indices = np.take(indices, stepsize_idx, axis=0)

    return indices


def im2col(A, indices):
    return np.take(A, indices)


def col2im(input, original_shape, original_indecses):
    output = np.zeros(original_shape).flatten()
    np.add.at(output, original_indecses.flatten(), input.flatten())
    return output.reshape(original_shape)
