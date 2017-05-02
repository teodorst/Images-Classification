import numpy as np


def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    M,N, _ = A.shape
    print M, N
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    print start_idx
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    print offset_index
    print offset_idx.ravel()[:,None]
    # Get all actual indices & index into input array for final output
    print start_idx.ravel()
    print offset_idx.ravel()

    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])


def im2col(image, block_size):

    # height sa fie 3 !!!
    print image.shape
    rows, cols, height = image.shape
    horz_blocks = cols - block_size[1] + 1
    vert_blocks = rows - block_size[0] + 1

    print horz_blocks
    print vert_blocks

    output_vectors = np.zeros((block_size[0] * block_size[1]* height, horz_blocks * vert_blocks))
    itr = 0
    for v_b in xrange(vert_blocks):
        for h_b in xrange(horz_blocks):

            #image[v_b: v_b + block_size[0], h_b: h_b + block_size[1]].ravel()
            output_vectors[:, itr] = image[v_b: v_b + block_size[0], h_b: h_b + block_size[1]].ravel()
            itr += 1

    return output_vectors

def m_im2col(A, size, stepsize=1):
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
    print 'start_idx'
    print(start_idx.ravel())

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    print 'offset_idx'
    print(offset_idx.ravel()[:,None])
    print 'Output indeces'
    indices = (offset_idx.ravel()[:,None] + start_idx.ravel()).T
    print indices

    # Get all actual indices & index into input array for final output
    print 'Output values'
    output = np.take(A, indices)

    return output, indices

# def col2im(input, original_shape, original_indecses):
#     output = np.zeros(original_shape)
#     x, y = original_indecses.shape
#     for i in xrange(x):
#       np.put(output, original_indecses[i], input[i])

#     return output

def col2im(input, original_shape, original_indecses):
    output = np.zeros(original_shape)
    np.put(output, original_indecses.ravel(), input.ravel())
    return output.reshape(original_shape)


# def m_im2col(image, BSZ):
  # # x = np.arange(0, 4*4*3)
  # # print 'x', x
  # # print np.take(x, [0, 2])
  # # BSZ = (2, 2)
  # C, M, N = image.shape
  # # M = 4
  # # N = 4
  # # C = 3
  # stepsize = 1
  # col_extent = N - BSZ[1] + 1
  # row_extent = M - BSZ[0] + 1


  # start_idx = np.arange(BSZ[0])[:,None]*M + np.arange(BSZ[1])

  # offset_idx = np.arange(row_extent)[:,None] + np.arange(col_extent)

  # print 'start_idx', start_idx
  # print 'offset:', offset_idx
  # print 'offset ravel cols:', offset_idx.ravel()[:,None]
  # output = (offset_idx.ravel()[:,None] + start_idx.ravel().repeat(C))
  # print output
  # output = start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize]
  # print output
  # # return output + np.tile(np.arange(C), (row_extent*row_extent, BSZ[1]*BSZ[1]))



if __name__ == '__main__':
  print im2col(np.arange(0, 4*4*3).reshape((4,4,3)), (2,2))





# if __name__ == '__main__':
#   x = np.arange(0, 4*4*3)
#   print 'x', x
#   # print np.take(x, [0, 2])
#   BSZ = (2, 2)
#   M = 4
#   N = 4
#   C = 3
#   stepsize = 1
#   col_extent = N - BSZ[1] + 1
#   row_extent = M - BSZ[0] + 1

#   print col_extent
#   print row_extent

#   start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])

#   offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
#   print '1'
#   print np.arange(row_extent)[:,None]*N
#   print '2'
#   print start_idx.ravel()
#   print  np.tile(np.array([0, -1, -2]), C)
#   print offset_idx
#   output = (offset_idx.ravel()[:,None] + start_idx.ravel().repeat(C))*C
#   print output + np.tile(np.array([0, 1, 2]), (row_extent*row_extent, BSZ[1]*BSZ[1]))
#   # print offset_idx.ravel()[:,None] + start_idx.ravel()
#   # print np.take (x, offset_idx.ravel()[:,None] + start_idx.ravel().)

#   # print 'im2col'
#   # print im2col_sliding_broadcasting(x, (2, 2))
