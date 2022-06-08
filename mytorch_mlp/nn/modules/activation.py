import numpy as np
import math


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A =  1/(1 + np.exp(-Z)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A*(1-self.A) # TODO
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.tanh(Z) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - (self.A**2) # TODO
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = Z*(Z>0)
        
        return self.A
    
    def backward(self):
        dAdZ = np.zeros((np.shape(self.A)[0],np.shape(self.A)[1]))
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if self.A[i][j] == 0:
                    dAdZ[i][j] = 0
                else:
                    dAdZ[i][j] = 1 
        return dAdZ
        
        
