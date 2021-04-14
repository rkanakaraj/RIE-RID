from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import sympy
from scipy.integrate import odeint
import imageio

def vertical_split(img, left, top, right, bottom):
    """
    crops the image based on the given co-ordinates.
    Co-ordinates are of the order left, top, right, bottom
    """
    return img.crop((left, top, right, bottom))

def zig_zag_scanning(arr):
    """
    returns the zig-zag traversal of the input array
    right, down, right, up order
    input- a: 2d list whose zig-zag pattern should be got
    output- 1d list with zig-zag traversed elements
    """
    rows = len(arr)
    columns = len(arr[0])
        
    temp = [[] for i in range(rows+columns-1)]
      
    for i in range(rows):
        for j in range(columns):
            x = i+j
            if(x%2 ==0):
                temp[x].insert(0,arr[i][j])
            else:
                temp[x].append(arr[i][j])
    #zig-zag traversal
    zig_zag = []
    for i in temp:
        for j in i:
            zig_zag.append(j)
    return zig_zag

def spiral_scanning(a):
    """
    returns the spiral traversal of the input array
    input- a: 2d list whose spiral pattern should be got
    output- 1d list with spirally traversed elements
    """
    m = len(a)
    n = len(a[0])

    b = []

    i, k, l = 0, 0, 0
  
    # Total elements in matrix
    size = m * n
  
    while (k < m and l < n):  
        for i in range(l, n):
            b.append(a[k][i])
        k += 1
  
        # Print the last column
        # from the remaining columns
        for i in range(k, m):
            b.append(a[i][n - 1])  
        n -= 1
  
        # Print the last row 
        # from the remaining rows
        if (k < m):
            for i in range(n - 1, l - 1, -1):
                b.append(a[m - 1][i])  
        m -= 1
  
        # Print the first column 
        # from the remaining columns 
        if (l < n):
            for i in range(m - 1, k - 1, -1):
                b.append(a[i][l])
            l += 1
  
    return b[::-1]

def rearrange(arr, n, m):
    """
    rearranges 1d array into a matrix of the given dimension
    input- arr: 1d array, n:number of rows, m:number of columns
    output: 2d list of array values
    """
    
    matrix = [[0 for i in range(m)] for j in range(n)]
    z=0
    for i in range(n):
        for j in range(m):
            matrix[i][j] = arr[z]
            z+=1
    return matrix

def flip_and_merge(mat1, mat2, n, m):
    """
    flip the given two matrices upside down and merge them by alernating columns
    input- mat1, mat2: matrices to be merged, n,m: dimensions of the matrices
    output- merged matrix of dimension n x 2m
    """
    
    merged_matrix = [[0 for i in range(2*m)] for j in range(n)]
    x = 0
    for j in range(m):
        for i in range(n-1, -1, -1):
            merged_matrix[i][x] = mat1[n-i-1][j] 
        x = x+1
        
        for k in range(n-1, -1, -1):
            merged_matrix[k][x] = mat2[n-k-1][j]
        x = x+1
    return merged_matrix

#TODO: move to utils
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
######################

def discrete_logarithm(Zp, p, phi_n, a=None):
  dis_log, roots, order = [], [], None
  for g in Zp:
    row = []
    for x in range(1, phi_n+1):
      val = pow(int(g), int(x), p)
      row.append(val)
    for j in row:
      if j == 1:
        if a:
          order = row.index(j)+1
        if row.index(j)+1 == phi_n:
          roots.append(g)
        break
    dis_log.append(row)
  return roots[0]

def generate_Zn(n):
  Zn = []
  for i in range(1, n):
    if math.gcd(i, n) == 1:
      Zn.append(i)
  return Zn

def generate_elgamal_key():
    while True:
        try:
            p = sympy.randprime(300, 1080)
            Zp = generate_Zn(p)
            d = random.randint(1, p-2)
            e1 = discrete_logarithm(Zp, p, len(Zp))
            break
        except Exception as e:
            print("Error", e)
            continue
    e2 = pow(e1, d, p)
    return e1,e2,p,d,Zp

def encrypt(e1, e2, p, Zp, message_matrix, r):
    """
    elgamal
    encrypts the message 
    input- e1, e2, p: public keys, message_matrix: matrix to be encrypted
    output- encrypted matrix
    """
    
    c1 = pow(e1, r, p)
    temp = pow(e2, r, p)
    
    encrypted = np.asarray([np.asarray([0 for i in range(len(message_matrix[0]))])
                            for j in range(len(message_matrix))])
    for i in range(len(message_matrix)):
        for j in range(len(message_matrix[0])):
            encrypted[i][j] = ((message_matrix[i][j]*temp)%p)
    return encrypted, c1
           
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

def encrypt_lchaos(matrix, order):
    new = np.ones(matrix.shape)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = matrix[i][order[i][j]]
    return new

def encrypt_rchaos(matrix, x):
    G = np.ones(matrix.shape)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j!= 0:
                G[i][j] = int(G[i][j-1])^int(matrix[i][j])^int(x[i][j])
            else:
                G[i][j] = 0^int(matrix[i][j])^int(x[i][j])
    return G
    

img = Image.open("../src/test.png").convert('LA')
width, height = img.size
img = img.resize((width-width%2,height-height%2))
width, height = img.size
plt.imshow(img)
plt.show()

"""
STEP 1
splitting image
"""

#left half
left = 0
top = 0
right = width//2
bottom = height

img1 = vertical_split(img, left, top, right, bottom)
print("first half:\n")
plt.imshow(img1)
plt.show()


#right half
left = width//2
right = width
img2 = vertical_split(img, left, top, right, bottom)
print("second half:\n")
plt.imshow(img2)
plt.show()

#converting to array of pixels
img1_arr = mat_to_arr(np.asarray(img1))
img2_arr = mat_to_arr(np.asarray(img2))

broken_height, broken_width = img1_arr.shape[0], img1_arr.shape[1]

"""
STEP 2
zig zag scanning
"""

zig_zag_scanned = zig_zag_scanning(img1_arr)


"""
STEP 3
spiral scanning
"""

spiral_scanned = spiral_scanning(img2_arr)

"""
STEP 4
arrange traversed lists back to matrix form
"""

shuff1 = rearrange(zig_zag_scanned, broken_height, broken_width )
print("Zig-zag shuffled:")
z = Image.fromarray(arr_to_mat(np.asarray(shuff1)))
plt.imshow(z)
plt.show()
print("spiral shuffled:")
shuff2 = rearrange(spiral_scanned, broken_height, broken_width )
s = Image.fromarray(arr_to_mat(np.asarray(shuff2)))
plt.imshow(s)
plt.show()

"""
STEP 5
merge the broken matrices by alternate columns from both matrices
"""

merged_matrix = flip_and_merge(shuff1, shuff2, broken_height, broken_width)
print("Merged and flipped:")
m = Image.fromarray(arr_to_mat(np.asarray(merged_matrix)))
plt.imshow(m)
plt.show()

"""
STEP 6
generate keys for elgamal cyptosystem and encrypt the matrix
"""

e1,e2,p,d,Zp = generate_elgamal_key()
z = random.choice(Zp)
encrypted_matrix, c1 = encrypt(e1, e2, p, Zp, merged_matrix, z)
# np.savetxt("../decryption/matrix.txt", encrypted_matrix, fmt='%.2f')
print("Elgamal encrypted:")

e = Image.fromarray(arr_to_mat(np.asarray(encrypted_matrix)))
plt.imshow(e)
plt.show()

with open("../decryption/key","w") as f:
    f.write(str(d)+' '+str(c1)+' '+str(p))


# e.save("../decryption/encrypted.png")

"""
STEP 7
Lorentz chaos system encryption
"""
x = generate_lorentz(encrypted_matrix.size).transpose()[0]
order = np.argsort(np.reshape(x, encrypted_matrix.shape))
lorentz_encrypted_matrix = encrypt_lchaos(encrypted_matrix, order)
print("Lorenz encrypted:")
#f = Image.fromarray(arr_to_mat(np.asarray(encrypted_matrix)))

f = imageio.core.util.Array(lorentz_encrypted_matrix)
_ = plt.imshow(f,"Greys")
plt.show()

#imageio.imwrite("../decryption/matrix.tiff", f)

"""
STEP 8
Rossler chaos system encryption
"""

r_x = generate_rossler(lorentz_encrypted_matrix.size).transpose()[0]
r_x = np.resize(r_x, lorentz_encrypted_matrix.shape)
encrypted_matrix = encrypt_rchaos(lorentz_encrypted_matrix, r_x)
final_result = imageio.core.util.Array(lorentz_encrypted_matrix)
print("Rossler encrypted:")
_ = plt.imshow(final_result,"Greys")

imageio.imwrite("../decryption/matrix.tiff", final_result)