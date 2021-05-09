from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio

from crypto_models.scanning_systems import ZigZagCS, SpiralCS
from crypto_models.utils import vertical_split, mat_to_arr, arr_to_mat, flip_and_merge
from crypto_models.elgamal import ElgamalCS
from crypto_models.chaos_systems import LorenzCS, RosslerCS

class Encrypt:
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
        print("Encryption report:")
        self.plot(self.org_mat, "Original image")
        self.plot([self.l_mat, self.r_mat], ["before zigzag", "before spiral"], [1,2])
        self.plot([self.el_mat, self.er_mat], ["after zigzag", "after spiral"], [1,2])
        self.plot(self.m_mat, "Merged Matrix")
        self.plot(self.ee_mat, "Elgamal encrypted matrix")
        self.plot(self.lc_mat, "Lorenz encrypted matrix")
        self.plot(self.rc_mat, "Rossler encrypted matrix")
        self.plot([self.org_mat, self.rc_mat],["initial matrix","final encrpyted matrix"],[1,2])
        
    def encrypt(self, img, sav_keys="../dump/key", sav_file="../dump/matrix.tiff", save=True):
        """
            Encrypt according to the paper.
        """
        try:
            self.org_mat = mat_to_arr(np.array(img))
        except:
            self.org_mat = np.array(img)
        """
        STEP 1
        splitting image
        """
        self.l_mat, self.r_mat = vertical_split(self.org_mat)
        print("Image split")
        
        """
        STEP 2
        zig zag scanning
        """
        z = ZigZagCS()
        zig_zag_scanned = z.encrypt(self.l_mat)
        print("Zigzag scanned")
        
        """
        STEP 3
        spiral scanning
        """
        s = SpiralCS()
        spiral_scanned = s.encrypt(self.r_mat)
        print("Spiral scanned")
        
        """
        STEP 4
        arrange traversed lists back to matrix form
        """
        self.el_mat = np.resize(zig_zag_scanned, self.l_mat.shape)
        self.er_mat = np.resize(spiral_scanned, self.r_mat.shape)
        
        """
        STEP 5
        merge the broken matrices by alternate columns from both matrices
        """
        self.m_mat = flip_and_merge(self.el_mat, self.er_mat, *self.org_mat.shape)
        print("Image merged and flipped")
        
        """
        STEP 6
        generate keys for elgamal cyptosystem and encrypt the matrix
        """
        e = ElgamalCS()
        e.make_key()
        e1,e2,p,d,Zp = e.get_keys()
        self.ee_mat, c1 = e.encrypt(self.m_mat)
        print("Elgamal encrypted")
        
        with open(sav_keys,"w") as f:
            f.write(str(d)+' '+str(c1)+' '+str(p))
        print("Keys saved")
        
        """
        STEP 7
        Lorentz chaos system encryption
        """
        l = LorenzCS()
        self.lc_mat, x = l.encrypt(self.ee_mat)
        print("Lorenz encrypted")
        
        """
        STEP 8
        Rossler chaos system encryption
        """
        r = RosslerCS()
        self.rc_mat,x = r.encrypt(self.lc_mat)
        print("Rossler encrypted")
        
        final_result = imageio.core.util.Array(self.rc_mat)
        
        if save:
            imageio.imwrite(sav_file, final_result)
            print("Image saved")
        
        return final_result

e = Encrypt()
img = np.array(Image.open("../image/sample2.jpg")) # .convert('LA')
key_const = "../dump/key"
sav_file = "../dump/matrix.tiff"

# grey scale
if len(img.shape)==2:
    result = e.encrypt(img)
    e.plot_stat()
# color immage
else:
    f_map = []
    plt.imshow(img)
    plt.title("Original image")
    plt.show()
    for i in range(img.shape[2]):
        print("\nChannel %d:"%i)
        f_map.append(e.encrypt(img[:, :, i], sav_keys=key_const+str(i), save=False))
        e.plot_stat()
    res = np.ones(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j] = [f_map[k][i][j] for k in range(3)]
    imageio.imwrite(sav_file, res)
    plt.imshow(res.astype(np.uint8))
    plt.title("Encrypted image")
    plt.show()
    print("Image saved")