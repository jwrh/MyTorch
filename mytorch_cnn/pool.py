import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        
        """
        self.idx = np.empty((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1,2),dtype = np.int8)
        Z = np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1))
        self.shape = A.shape
        for a in range(Z.shape[0]):
            for b in range(Z.shape[1]):
                for c in range(Z.shape[2]):
                    for d in range(Z.shape[3]):
                        k = A[a,b,c:c+self.kernel,d:d+self.kernel]
                        x = np.where(k == np.amax(k))[0][0]
                        y = np.where(k == np.amax(k))[1][0]
                        self.idx[a,b,c,d,0] = c+x
                        self.idx[a,b,c,d,1] = d+y
                        Z[a,b,c,d] = np.amax(k)
        return Z 
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.shape)
        self.sha = dLdZ.shape
        # print(self.sha)
        for a in range(self.sha[0]):
            for b in range(self.sha[1]):
                for c in range(self.sha[2]):
                    for d in range(self.sha[3]):
                        x =self.idx[a,b,c,d][0]
                        y =self.idx[a,b,c,d][1]
                        dLdA[a,b,x,y] += dLdZ[a,b,c,d]
        print(self.sha)
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1))
        self.shape = A.shape
        self.ksq = (self.kernel**2)
        for a in range(Z.shape[0]):
            for b in range(Z.shape[1]):
                for c in range(Z.shape[2]):
                    for d in range(Z.shape[3]):
                        k = A[a,b,c:c+self.kernel,d:d+self.kernel]
                        n = np.sum(k,axis = (0,1))/self.ksq
                        Z[a,b,c,d] = n
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.shape)
        self.sha = dLdZ.shape
        # print(self.sha)
        for c in range(self.sha[2]):
            for d in range(self.sha[3]):
                for i in range(self.kernel):
                    for j in range(self.kernel):
                        dLdA[:,:,c+i,d+j] += dLdZ[:,:,c,d]/self.ksq
        print(self.sha)
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride) 

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)#TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA
