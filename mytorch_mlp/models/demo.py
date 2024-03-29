from re import I
import numpy as np

from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU

class MLP0:

    def __init__(self, debug = False):
    
        self.layers = [ Linear(2, 3) ]
        self.f      = [ ReLU() ]

        self.debug = debug

    def forward(self, A0):

        Z0 = np.dot(A0,self.layers[0].W.T) + np.dot(np.ones((A0.shape[0],1), dtype="f"),self.layers[0].b.T)   
        A1 = self.f[0].forward(Z0)   
        self.A0 = A0
        self.N = A0.shape[0] 
        u = self.layers[0].forward(A0)
        if self.debug:

            self.Z0 = Z0
            self.A1 = A1
        
        return A1

    def backward(self, dLdA1):
    
        dA1dZ0 = self.f[0].backward()   
        dLdZ0  = np.multiply(dLdA1,dA1dZ0)   
        dLdA0  = np.dot(dLdZ0,self.layers[0].W)   
        u = self.layers[0].backward(dLdZ0)
        dLdW0 = self.layers[0].dLdW
        dLdb0 = self.layers[0].dLdb
        if self.debug:

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
            self.dLdW0 = dLdW0
            self.dLdb0 = dLdb0
        
        return None
        
class MLP1:

    def __init__(self, debug = False):
    
        self.layers = [ Linear(2, 3),
                        Linear(3, 2) ]
        self.f      = [ ReLU(),
                        ReLU() ]

        self.debug = debug

    def forward(self, A0):
        Z0 = np.dot(A0,(self.layers[0].W.T))+np.dot(np.ones((A0.shape[0],1), dtype="f"), self.layers[0].b.T)   
        A1 = self.f[0].forward(Z0)   
        x = self.layers[0].forward(A0)   
        Z1 = np.dot(A1,self.layers[1].W.T)+np.dot(np.ones((A0.shape[0],1), dtype="f"), self.layers[1].b.T)    
        A2 = self.f[1].forward(Z1)   

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2
        
        return A2

    def backward(self, dLdA2):

        dA2dZ1 = self.f[1].backward()  
        dLdZ1  = np.multiply(dLdA2,dA2dZ1)    
        dLdA1  = np.dot(dLdZ1,self.layers[1].W)   
        dA1dZ0 = self.f[0].backward()   
        dLdZ0  =  np.multiply(dLdA1,dA1dZ0)    
        dLdA0  = np.dot(dLdZ0,self.layers[0].W)   
        u = self.layers[0].backward(dLdZ0)
        if self.debug:

            self.dA2dZ1 = dA2dZ1
            self.dLdZ1  = dLdZ1
            self.dLdA1  = dLdA1

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
        
        return None

class MLP4:
    def __init__(self, debug=False):
        
        # Hidden Layers
        self.layers = [
            Linear(2, 4),
            Linear(4, 8),
            Linear(8, 8),
            Linear(8, 4),
            Linear(4, 2)]

        # Activations
        self.f = [
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU()]

        self.debug = debug

    def forward(self, A):

        if self.debug:

            self.Z = []
            self.A = [ A ]

        L = len(self.layers)

        for i in range(L):
            u = self.layers[i].forward(self.A[i])
            Z = np.dot(self.A[i],(self.layers[i].W.T))+np.dot(np.ones((self.A[i].shape[0],1), dtype="f"), self.layers[i].b.T)    
            A = self.f[i].forward(Z)   
            
            if self.debug:

                self.Z.append(Z)
                self.A.append(A)
        
            

        return A

    def backward(self, dLdA):

        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [ dLdA ]

        L = len(self.layers)

        for i in reversed(range(L)):
            dAdZ = self.f[i].backward()   
            dLdZ = np.multiply(dLdA,dAdZ)   
            dLdA = np.dot(dLdZ,self.layers[i].W)  
            u = self.layers[i].backward(dLdZ)   

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA

        return None