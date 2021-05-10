# utils
import numpy as np

def join(im1, im2):
    return np.transpose(np.asarray(list(np.transpose(im1))+list(np.transpose(im2))))

def vertical_split(img):
    """
    crops the image based on the given co-ordinates.
    Co-ordinates are of the order left, top, right, bottom
    """
    width, height = img.shape
    
    if width%2:
        width+=1
    img1 = img.transpose()[:width//2].transpose() 
    img2 = img.transpose()[width//2:].transpose() 
    
    
    return img1, img2

def arr_to_mat(arr):
    proc_image = []
    for i in range(len(arr)):
        temp = []
        for j in range(len(arr[0])):
            temp.append(np.asarray([arr[i][j],255]).astype(np.uint8))
        proc_image.append(np.asarray(temp))
    return np.asarray(proc_image)


def mat_to_arr(image):
    proc_image = []
    for i in range(len(image)):
        temp = []
        for j in range(len(image[0])):
            temp.append(image[i][j][0])
        proc_image.append(np.asarray(temp))
    return np.asarray(proc_image)

def flip_and_split(image):
    """
        flip the matrix and split the matrix into two matrices 
        split wrt alternating columns
        returns
            split matrices
    """
    image = np.transpose(image)
    im1, im2 = [],[]
    for i in range(len(image)):
        if i%2==0:
            im1.append(image[i][::-1])
        else:
            im2.append(image[i][::-1])
    im1, im2 = np.transpose(np.asarray(im1)), np.transpose(np.asarray(im2))
    
    return im1, im2

def flip_and_merge(mat1, mat2, w, h):
    """
        flip the given two matrices upside down and merge them by alernating columns
        input- mat1, mat2: matrices to be merged, w,h: dimensions of the result matrices
        returns
            merged matrix of dimension w x h
    """
    
    merged_matrix = np.ones((w,h))
    mat1 = mat1.transpose()
    mat2 = mat2.transpose()
    x = 0
    for j in range(w//2):
        merged_matrix[x] = mat1[j][::-1]
        x = x+1
        merged_matrix[x] = mat2[j][::-1]
        x = x+1
    if w%2:
        merged_matrix[-1] = mat1[-1][::-1]
    return merged_matrix.transpose()