from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import imageio
from scipy.integrate import odeint

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
    
    return im1, im2

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


def euclideanModInverse(a, m):  
    if a == 0 :   
        return m, 0, 1
    gcd, x1, y1 = euclideanModInverse(m%a, a)  
    x = y1 - (m//a) * x1  
    y = x1
    return gcd, x, y

def decrypt(image, secret, c, p):
    _, inv, _ = euclideanModInverse(pow(c, secret), p)
    new_image = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            new_image[i][j] = (image[i][j]*inv)%p
    return new_image


def generate_lorentz(num, sigma=10, beta=8/3, rho=28):
    """ 
        returns lorentz chaotic sequence of x,y,z
    """
    def f(state, t):
        """ returns Derivatives of x,y,z """
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  
    
    state0 = [1.0, 1.0, 1.0] #init
    t = np.arange(num)
    
    return odeint(f, state0, t)

    
def generate_rossler(num, sigma=0.2, beta=0.2, rho=5.7):
    """ 
        returns rossler chaotic sequence of x,y,z
    """
    def f(state, t):
        """ returns Derivatives of x,y,z """
        x, y, z = state
        return -y - z, x + sigma*y, beta + z*(x-rho)
    
    state0 = [1.0, 1.0, 1.0] #init
    t = np.arange(num)
    
    return odeint(f, state0, t)

def decrypt_rossler(matrix, x):
    new = np.ones(matrix.shape)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j!= 0:
                new[i][j] = int(new[i][j-1])^int(matrix[i][j])^int(x[i][j])
            else:
                new[i][j] = 0^int(matrix[i][j])^int(x[i][j])
    return new


def decrypt_lorenz(matrix, order):
    new = np.ones(matrix.shape)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][order[i][j]] = matrix[i][j]
    return new

org_img_arr = np.array(imageio.imread("matrix.tiff"))
width, height = org_img_arr.shape

plt.imshow(org_img_arr, "Greys")
plt.show()

# org_img_arr = np.loadtxt("../decryption/matrix.txt", dtype='i', delimiter=' ')

with open("../decryption/key") as f:
    secret, c, p = [int(i) for i in f.read().split()]


"""
STEP 8
inverse of
Rossler chaos system encryption
"""

# r_x = generate_rossler(org_img_arr.size).transpose()[0]
# r_x = np.resize(r_x, org_img_arr.shape)
# img_arr = decrypt_rossler(org_img_arr, r_x)
img_arr = org_img_arr
"""
STEP 7
inverse of
Lorentz chaos system encryption
"""

x = generate_lorentz(img_arr.size).transpose()[0]
order = np.argsort(np.reshape(x, img_arr.shape))
img_arr2 = decrypt_lorenz(img_arr, order)

"""
STEP 6
inverse of
elgamal cyptosystem 
"""

img_arr = decrypt(img_arr2, secret, c, p)

plt.imshow(Image.fromarray(arr_to_mat(img_arr)))
plt.show()

"""
STEP 5
inverse of 
merge the broken matrices by alternate columns from both matrices
"""

im1, im2 = flip_and_split(img_arr)
s1 = im1.shape
s2 = im2.shape
plt.imshow(Image.fromarray(arr_to_mat(im1)))
plt.show()
plt.imshow(Image.fromarray(arr_to_mat(im2)))
plt.show()


"""
STEP 4
inverse of 
arrange traversed lists back to matrix form
"""

im1, im2 = np.ravel(im1), np.ravel(im2)

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

