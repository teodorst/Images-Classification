import numpy as np
from util import close_enough


def im2col(A, size, stepsize=1):
    B = size
    skip = [1,1]
    # Parameters
    D,M,N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])
    # print(start_idx)

    # Generate Depth indeces
    didx=M*N*np.arange(D)
    # print(didx)

    start_idx=(didx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    indices = (offset_idx.ravel()[:,None] + start_idx.ravel())

    if stepsize > 1:
        stepsize_idx = []
        for i in range(indices.shape[0]):
            block = indices[i][0] // M
            if block % stepsize == 0 and i % stepsize == 0:
                stepsize_idx.append(i)
        indices = np.take(indices, stepsize_idx, axis=0)


    # Get all actual indices & index into input array for final output
    # print 'Output values'

    output = np.take(A, indices)
    return output, indices


def col2im(input, original_shape, original_indecses):
    output = np.zeros(original_shape).flatten()
    np.add.at(output, original_indecses.flatten(), input.flatten())
    return output.reshape(original_shape)


if __name__ == '__main__':
  x = np.arange(3* 32 * 32).reshape(3, 32, 32)
  im2col(x, (2,2), 2)
