# chaos systems - lorenz + rossler
import numpy as np
from scipy.integrate import odeint

class LorenzCS:
    def __init__(self, sigma=10, beta=8/3, rho=28):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
    
    def f(self, state, t):
        """ 
            returns 
                Derivatives of x,y,z - float (each)
        """
        x, y, z = state
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  
    
    def generate(self, seq_length):
        """ 
             returns lorentz chaotic sequence of x,y,z
             usage:
                 generate(matrix.size).transpose()[0] - to get x sequence
                 generate(matrix.size).transpose()[1] - to get y sequence
                 generate(matrix.size).transpose()[2] - to get z sequence
        """
        state0 = [1.0, 1.0, 1.0] # initial x, y, z
        t = np.arange(seq_length) # time indices
        
        return odeint(self.f, state0, t)

    def encrypt(self, matrix):
        """
            encrypts the input matrix based on the lorenz sequence.
            returns
                encrypted matrix - numpy 2d array
                lorenz x-sequence - list
        """
        x = self.generate(matrix.size).transpose()[0] # we need to consider only x sequence
        order = np.argsort(np.reshape(x, matrix.shape))
        encrypted_matrix = np.ones(matrix.shape)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                encrypted_matrix[i][j] = matrix[i][order[i][j]]
        return encrypted_matrix, x
    
    
    def decrypt(self, matrix):
        x = self.generate(matrix.size).transpose()[0] # we need to consider only x sequence
        order = np.argsort(np.reshape(x, matrix.shape))
        decrypted_matrix = np.ones(matrix.shape)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                decrypted_matrix[i][order[i][j]] = matrix[i][j]
        return decrypted_matrix
    
class RosslerCS:
    def __init__(self, sigma=0.2, beta=0.2, rho=5.7):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
    
    def f(self, state, t):
        """ 
            returns 
                Derivatives of x,y,z - float (each)
        """
        x, y, z = state
        return -y - z, x + self.sigma*y, self.beta + z*(x-self.rho)
    
    def generate(self, seq_length):
        """ 
            returns rossler chaotic sequence of x,y,z
            usage:
                 generate(matrix.size).transpose()[0] - to get x sequence
                 generate(matrix.size).transpose()[1] - to get y sequence
                 generate(matrix.size).transpose()[2] - to get z sequence
        """
        state0 = [1.0, 1.0, 1.0] # initial x,y,z
        t = np.arange(seq_length) # time indices
        
        return odeint(self.f, state0, t)
    
    def encrypt(self, matrix):
        """
            encrypts the input matrix based on the rossler sequence.
            returns
                encrypted matrix - numpy 2d array
                rossler x-sequence - list
        """
        x = self.generate(matrix.size).transpose()[0]
        order = np.resize(x, matrix.shape)
        encrypted_matrix = np.ones(matrix.shape)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if j!= 0:
                    encrypted_matrix[i][j] = int(encrypted_matrix[i][j-1])^int(matrix[i][j])^int(order[i][j])
                else:
                    encrypted_matrix[i][j] = 0^int(matrix[i][j])^int(order[i][j])
        return encrypted_matrix, x
    
    def decrypt(self, matrix):
        x = self.generate(matrix.size).transpose()[0] # we need to consider only x sequence
        x = np.resize(x, matrix.shape)
        decrypted_matrix = np.ones(matrix.shape)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if j!= 0:
                    decrypted_matrix[i][j] = int(matrix[i][j-1])^int(matrix[i][j])^int(x[i][j])
                else:
                    decrypted_matrix[i][j] = 0^int(matrix[i][j])^int(x[i][j])
        return decrypted_matrix
            