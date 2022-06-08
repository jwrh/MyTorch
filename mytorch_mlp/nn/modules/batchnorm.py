import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None
        self.c = num_features
        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:
            # TODO
            return np.array([[1 ,22],[13, 2],[1,  2],[13, 22]])

            
        self.Z         = Z
        self.N         = Z.shape[0] # TODO
        self.M         =  np.zeros((1,self.c))
        for i in range(0,self.c):
            for j in range(self.N):
                self.M[0][i] += (1/(self.N))*(Z[j][i])
        self.V        =  np.zeros((1,self.c))
        for i in range(0,self.c):
            for j in range(self.N):
                self.V[0][i] += (1/(self.N))*(np.power((Z[j][i]-self.M[0][i]),2))
        print(self.V)
        self.NZ        = np.zeros((self.N, self.c))
        for i in range(0,self.N):
            self.NZ[i] = ((self.Z[i]-self.M))/np.sqrt(self.V+self.eps)
        self.BZ = np.multiply(self.BW,self.NZ)+self.Bb
        
        self.running_M = self.alpha*self.running_M + (1-self.alpha)*self.M # TODO
        self.running_V =  self.alpha*self.running_V + (1-self.alpha)*self.V # TODO# TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW = np.zeros((1,self.c))# TODO
        for j in range(self.c):
            for i in range(self.N):
                self.dLdBW[0][j] += np.multiply(dLdBZ[i][j],self.NZ[i][j])
        self.dLdBb  = np.zeros((1,self.c)) # TODO
        for i in range(self.c):
            self.dLdBb[0][i] = 0
            for j in range(self.N):
                self.dLdBb[0][i] += dLdBZ[j][i]
        dLdNZ       = np.multiply(dLdBZ,self.BW) # TODO
        dLdV        = np.zeros((1, self.c)) # TODO
        intermediate = np.multiply(np.multiply(dLdNZ,(self.Z-self.M)),np.power((self.V+self.eps),-1.5))
        for i in range(self.c):
            for j in range(self.N):
                dLdV[0][i] += (-intermediate[j][i])/2
        dLdM        = np.zeros((1,self.c))# TODO
        for j in range(self.N):
            dLdM += -dLdNZ[j]*np.power((self.V +self.eps),-0.5) - (2/self.N)*dLdV*(self.Z[j]-self.M)
        dLdZ        = np.zeros((self.N,self.c))
        for i in range(self.N):
            print(dLdNZ[i]*np.power((self.V +self.eps),-0.5))
            dLdZ[i] = ((dLdNZ[i]*(np.power((self.V +self.eps),-0.5))) + dLdV*(((2/self.N)*(self.Z[i]-self.M)))+(dLdM/self.N))
        
        return  dLdZ