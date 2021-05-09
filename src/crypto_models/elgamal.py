# elgamal
import math 
import sympy 
import random
import numpy as np

class ElgamalCS:
    def __init__(self):
        pass
    
    def make_key(self):
        self.e1, self.e2, self.p, self.d, self.Zp = self.generate_elgamal_key()
    
    def get_keys(self):
        return self.e1, self.e2, self.p, self.d, self.Zp
    
    def generate_Zp(self, n):
        Zp = []
        for i in range(1, n):
            if math.gcd(i, n) == 1:
                Zp.append(i)
        return Zp
    
    def discrete_logarithm(self, Zp, p, phi_n, a=None):
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

    def generate_elgamal_key(self):
        """
            Generates elgamal keys
        """
        while True:
            try:
                p = sympy.randprime(300, 1080)
                Zp = self.generate_Zp(p)
                d = random.randint(1, p-2)
                e1 = self.discrete_logarithm(Zp, p, len(Zp))
                break
            except Exception as e:
                print("Error", e)
                continue
        e2 = pow(e1, d, p)
        return e1,e2,p,d,Zp
    
    def euclideanModInverse(self, a, m):
        """
            Find euclidean mod inverse
            return
                gcd, inverse, dummy
        """
        if a == 0 :   
            return m, 0, 1
        gcd, x1, y1 = self.euclideanModInverse(m%a, a)  
        x = y1 - (m//a) * x1  
        y = x1
        return gcd, x, y
    
    def encrypt(self, message_matrix):
        """
            elgamal encrypts the message using 
            e1, e2, p: public keys.
            returns
                encrypted matrix - matrix
                c1 - used for decryption - float
        """
        r = random.choice(self.Zp)

        c1 = pow(self.e1, r, self.p)
        temp = pow(self.e2, r, self.p)
        
        encrypted = np.asarray([np.asarray([0 for i in range(len(message_matrix[0]))])
                                for j in range(len(message_matrix))])
        for i in range(len(message_matrix)):
            for j in range(len(message_matrix[0])):
                encrypted[i][j] = ((message_matrix[i][j]*temp)%self.p)
        return encrypted, c1
    
    def decrypt(self, image, secret, c, p):
        """
            elgamal decrypt using keys.
            returns
                decrypted image - matrix
        """
        _, inv, _ = self.euclideanModInverse(pow(c, secret), p)
        new_image = [[0 for i in range(len(image[0]))] for j in range(len(image))]
        for i in range(len(image)):
            for j in range(len(image[0])):
                new_image[i][j] = (image[i][j]*inv)%p
        return new_image