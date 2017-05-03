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
    # print 'start_idx'
    # print(start_idx.ravel())

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    # print 'offset_idx'
    # print(offset_idx.ravel()[:,None])
    # print 'Output indeces'
    indices = (offset_idx.ravel()[:,None] + start_idx.ravel()).T
    # print indices

    # Get all actual indices & index into input array for final output
    # print 'Output values'
    output = np.take(A, indices)

    return output, indices


def col2im(input, original_shape, original_indecses):
    output = np.zeros(original_shape).flatten()
    np.add.at(output, original_indecses.flatten(), input.flatten())
    return output.reshape(original_shape)




def my_im2col(image, filter_size, stride = 1):
    depth, height, width = image.shape

    block_height = (height - filter_size[0]) / stride + 1
    block_width = (width - filter_size[1]) / stride + 1

    #print image.shape
    #print block_height, block_width
    #print filter_size

    index = 0
    output = np.zeros((filter_size[0] * filter_size[1] * depth, block_height * block_width))
    for h in range(0, block_height, stride):
      for w in range(0, block_width, stride):
        pixel_block = image[:, h : h + filter_size[0], w : w + filter_size[1]]
        output[:, index] = pixel_block.ravel()
        index += 1

    return output, None

def col2im_good2(im2col_mat, bsz, img_sz, stride = 1):

    b_rows, b_cols = bsz

    img_depth, img_rows, img_cols = img_sz
    # if img_depth != 3:
    #     print "Eroare prostule"
    #     return None
    img = np.zeros((img_depth, img_rows, img_cols))

    start_idx = []
    for row in range(0, img_rows - b_rows + 1, 1):
        for col in range(0, img_cols - b_cols + 1, 1):
            if row % stride == 0 and col % stride == 0:
                start_idx.append((row, col))

    print start_idx
    im2col_mat_row = 0
    for s_idx in start_idx:
        row, col = s_idx

        im2col_mat_col = 0
        for br in range(b_rows):
            for bc in range(b_cols):
                for crt_depth in range(img_depth):
                    img[crt_depth][row+br][col+bc] = im2col_mat[im2col_mat_row]\
                            [im2col_mat_col + b_cols*b_rows * crt_depth]
                im2col_mat_col += 1
        im2col_mat_row += 1
    return img

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
