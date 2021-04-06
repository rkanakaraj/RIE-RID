from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

def mat_to_arr(image):
    proc_image = []
    for i in range(len(image)):
        temp = []
        for j in range(len(image[0])):
            temp.append(image[i][j][0])
        proc_image.append(np.asarray(temp))
    return np.asarray(proc_image)

def arr_to_mat(arr):
    proc_image = []
    for i in range(len(arr)):
        temp = []
        for j in range(len(arr[0])):
            temp.append(np.asarray([arr[i][j],255]).astype(np.uint8))
        proc_image.append(np.asarray(temp))
    return np.asarray(proc_image)

def flip_and_split(image):
    image = np.transpose(image)
    im1, im2 = [],[]
    for i in range(len(image)):
        if i%2==0:
            im1.append(image[i][::-1])
        else:
            im2.append(image[i][::-1])
    im1, im2 = np.transpose(np.asarray(im1)), np.transpose(np.asarray(im2))
    # add alpha
    return arr_to_mat(im1), arr_to_mat(im2)

def inv_zig_zag(arr, shape):
    ct = 0
    flag = 0
    zigzag = np.ones(list(shape)[:2])
    for line in range(1, (shape[0] + shape[1])):
        start_col = max(0, line - shape[0])
        count = min(line, (shape[1] - start_col), shape[0])
        if flag == 0:
            for j in range(0, count):
                zigzag[min(shape[0], line)-j-1][start_col+j] = arr[ct].astype(np.uint8)
                ct += 1
            flag = 1
        else:
            for j in range(count-1, -1, -1):
                zigzag[min(shape[0], line)-j-1][start_col+j] = arr[ct].astype(np.uint8)
                ct += 1
            flag = 0
    return zigzag

def inv_spiral_scan(arr, shape):
    m = shape[0]
    n = shape[1]
    k = 0
    l = 0
    ct = 0
    arr = arr[::-1]
    spiral = np.ones(list(shape)[:2])
 
    while (k < m and l < n):
        for i in range(l, n):
            spiral[k][i] = arr[ct]
            ct += 1
        k += 1
        
        for i in range(k, m):
            spiral[i][n - 1] = arr[ct]
            ct += 1
        n -= 1

        if (k < m):
            for i in range(n - 1, (l - 1), -1):
                spiral[m - 1][i] = arr[ct]
                ct += 1 
            m -= 1
 
        if (l < n):
            for i in range(m - 1, k - 1, -1):
                spiral[i][l] = arr[ct]
                ct += 1
            l += 1
    return spiral

def join(im1, im2):
    return np.transpose(np.asarray(list(np.transpose(im1))+list(np.transpose(im2))))

img = Image.open("encrypted.png").convert('LA')
width, height = img.size


plt.imshow(img)

img_arr = mat_to_arr(np.asarray(img))

"""
STEP 5
inverse of 
merge the broken matrices by alternate columns from both matrices
"""

im1, im2 = flip_and_split(img_arr)
s1 = im1.shape
s2 = im2.shape
plt.imshow(Image.fromarray(np.asarray(im1)))
plt.show()
plt.imshow(Image.fromarray(np.asarray(im2)))
plt.show()


"""
STEP 4
inverse of 
arrange traversed lists back to matrix form
"""

im1, im2 = np.ravel(mat_to_arr(im1)), np.ravel(mat_to_arr(im2))

"""
STEP 3
inverse of 
zig zag scanning
"""

inv_zz = inv_zig_zag(im1,s1)
plt.imshow(Image.fromarray(arr_to_mat(inv_zz)))
plt.show()


"""
STEP 2
inverse of 
spiral scanning
"""

inv_ss = inv_spiral_scan(im2,s2)
plt.imshow(Image.fromarray(arr_to_mat(inv_ss)))
plt.show()


"""
STEP 1
inverse of 
splitting image
"""

dec_image = join(inv_zz, inv_ss)
plt.imshow(Image.fromarray(arr_to_mat(dec_image)))
plt.show()

