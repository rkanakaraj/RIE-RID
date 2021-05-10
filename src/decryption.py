from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import imageio
from scipy.integrate import odeint

from crypto_models.scanning_systems import ZigZagCS, SpiralCS
from crypto_models.utils import *
from crypto_models.elgamal import ElgamalCS
from crypto_models.chaos_systems import LorenzCS, RosslerCS


class Decrypt:
    def plot(self, mat, title, subplot=False):
        if not subplot:
            plt.imshow(Image.fromarray(arr_to_mat(mat)))
            plt.title(title)
        else:
            for i in range(len(title)):
                plt.subplot(subplot[0], subplot[1], i+1)
                plt.title(title[i])
                plt.imshow(Image.fromarray(arr_to_mat(mat[i])))
        plt.show()
        
    def plot_stat(self):
        print("Decryption report:")
        self.plot(self.org_mat, "Encrypted image")
        self.plot(self.lc_mat, "Lorenz decrypted matrix")
        self.plot(self.ee_mat, "Elgamal decrypted matrix")
        self.plot(self.rc_mat, "Rossler decrypted matrix")
        self.plot([self.l_mat, self.r_mat], ["before zigzag decryption", "before spiral decryption"], [1,2])
        self.plot([self.s_mat, self.z_mat], ["after zigzag decryption", "after spiral decryption"], [1,2])
        self.plot(self.res_mat, "Merged Matrix")
        self.plot([self.org_mat, self.res_mat],["initial matrix","final decrypted matrix"],[1,2])
        
    def decrypt(self, img, keys):
        width, height = img.shape
        self.org_mat = img
        
        """
        STEP 8
        inverse of
        Rossler chaos system encryption
        """
        r = RosslerCS()
        self.rc_mat = r.decrypt(img)
        print("Rossler decrypted")
        
        """
        STEP 7
        inverse of
        Lorentz chaos system encryption
        """
        l = LorenzCS()
        self.lc_mat = l.decrypt(self.rc_mat)
        print("Lorenz decrypted")
        
        """
        STEP 6
        inverse of
        elgamal cyptosystem 
        """
        e = ElgamalCS()
        self.ee_mat = e.decrypt(self.lc_mat, *keys)
        print("Elgamal decrypted")

        """
        STEP 5
        inverse of 
        merge the broken matrices by alternate columns from both matrices
        """
        self.l_mat, self.r_mat = flip_and_split(self.ee_mat)
        s1 = self.l_mat.shape
        s2 = self.r_mat.shape
        print("Split")
        
        """
        STEP 4
        inverse of 
        arrange traversed lists back to matrix form
        """
        im1, im2 = np.ravel(self.l_mat), np.ravel(self.r_mat)
        
        """
        STEP 3
        inverse of 
        zig zag scanning
        """
        z = ZigZagCS()
        self.z_mat = z.decrypt(im1,s1)
        print("Zigzag decrypted")
        
        """
        STEP 2
        inverse of 
        spiral scanning
        """
        s = SpiralCS()
        self.s_mat = s.decrypt(im2,s2)
        print("Spiral decrypted")
        
        """
        STEP 1
        inverse of 
        splitting image
        """
        self.res_mat = join(self.z_mat, self.s_mat)
        print("Merged matrix")
        return self.res_mat


d = Decrypt()
img = np.array(imageio.imread("../dump/matrix.tiff"))

key_const = "../dump/key"

if len(img.shape)==2:
    result = d.decrypt(img)
    d.plot_stat()
else:
    f_map = []
    plt.imshow(img.astype(np.uint8))
    plt.title("Encrpyted image")
    plt.show()
    for i in range(img.shape[2]):
        print("\nChannel %d:"%i)
        with open("../dump/key%d"%i) as f:
            keys = [int(i) for i in f.read().split()]
        f_map.append(d.decrypt(img[:, :, i], keys))
        d.plot_stat()
    res = np.ones(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j] = [f_map[k][i][j] for k in range(3)]
    plt.imshow(res.astype(np.uint8))
    plt.title("Decrypted image")
    plt.show()
    # imageio.imwrite(sav_file, res)
    print("Image decrypted")