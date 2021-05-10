# zig zag + spiral cryptosystem
import numpy as np

class ZigZagCS:
    def encrypt(self, mat):
        """
            Get the zig-zag traversal of the input matrix
            in right, down, right, up order
            returns
                zig-zag traversed elements - 1d list
        """
        rows = len(mat)
        columns = len(mat[0])
            
        temp = [[] for i in range(rows+columns-1)]
          
        for i in range(rows):
            for j in range(columns):
                x = i+j
                if(x%2 ==0):
                    temp[x].insert(0,mat[i][j])
                else:
                    temp[x].append(mat[i][j])
        #zig-zag traversal
        zig_zag = []
        for i in temp:
            for j in i:
                zig_zag.append(j)
        return zig_zag
    
    def decrypt(self, entries, shape):
        """
            Reverts the zigzag traversal entries to original matrix
            returns
                original matrix - 2d numpy array
        """
        ct = 0
        flag = 0
        zigzag = np.ones(list(shape)[:2])
        for line in range(1, (shape[0] + shape[1])):
            start_col = max(0, line - shape[0])
            count = min(line, (shape[1] - start_col), shape[0])
            if flag == 0:
                for j in range(0, count):
                    zigzag[min(shape[0], line)-j-1][start_col+j] = entries[ct].astype(np.uint8)
                    ct += 1
                flag = 1
            else:
                for j in range(count-1, -1, -1):
                    zigzag[min(shape[0], line)-j-1][start_col+j] = entries[ct].astype(np.uint8)
                    ct += 1
                flag = 0
        return zigzag

class SpiralCS:
    def encrypt(self, mat):
        """
            Get the spiral traversal of the input matrix
            returns
                spirally traversed elements - 1d list
        """
        m = len(mat)
        n = len(mat[0])
    
        b = []
    
        i, k, l = 0, 0, 0
        
        while (k < m and l < n):  
            for i in range(l, n):
                b.append(mat[k][i])
            k += 1
      
            # Print the last column
            # from the remaining columns
            for i in range(k, m):
                b.append(mat[i][n - 1])  
            n -= 1
      
            # Print the last row 
            # from the remaining rows
            if (k < m):
                for i in range(n - 1, l - 1, -1):
                    b.append(mat[m - 1][i])  
            m -= 1
      
            # Print the first column 
            # from the remaining columns 
            if (l < n):
                for i in range(m - 1, k - 1, -1):
                    b.append(mat[i][l])
                l += 1
      
        return b[::-1]
    
    def decrypt(self, arr, shape):
        """
            Reverts the spiral traversal entries to original matrix
            returns
                original matrix - 2d numpy array
        """
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
        
#TODO: get more work from parents