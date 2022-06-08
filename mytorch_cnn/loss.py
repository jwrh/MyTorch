import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     =np.multiply(np.subtract(self.A, self.Y),np.subtract(self.A, self.Y)) # TODO
        sse    = np.sum(se) # TODO
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = np.subtract(self.A, self.Y)
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = np.exp(self.A)/np.dot(np.dot(np.exp(self.A),Ones_C),Ones_C.T) # TODO
        crossentropy     = np.multiply(-self.Y, np.log(self.softmax)) # TODO
        sum_crossentropy = np.dot(np.dot(Ones_N.T, crossentropy),Ones_C)#None # TODO
        L = sum_crossentropy / N
        
        return L

    def backward(self):
    
        dLdA = np.subtract(self.softmax, self.Y)  # TODO
        
        return dLdA