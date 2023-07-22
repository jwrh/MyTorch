import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        k1 = A.shape[2]*self.upsampling_factor - (self.upsampling_factor-1)
        Z = np.zeros((A.shape[0],A.shape[1],k1))
        Z[:,:,::(self.upsampling_factor)] = A
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        dLdA =  dLdZ[:,:,::(self.upsampling_factor)]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        n  = A.shape[2]
        number = (A.shape[2]+(self.downsampling_factor-1))//(self.downsampling_factor)
        self.boolean  = A.shape[2]
        Z = np.zeros((A.shape[0],A.shape[1],number))
        for a in range(Z.shape[0]):
            for b in range(Z.shape[1]):
                counter  = 0
                for i in range(A.shape[2]):
                    if i%self.downsampling_factor == 0:
                        Z[a][b][counter] = A[a][b][i] 
                        counter += 1
        dLdA =  Z  
        return dLdA

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        n = dLdZ.shape[2]
        a = dLdZ.shape[0]
        b = dLdZ.shape[1]
        nf = self.boolean
        Z = np.zeros((a,b,nf))
        for n in range(a):
            for c in range(b):
                counter = 0
                for x in range(nf):
                    if x%self.downsampling_factor == 0:
                        Z[n][c][x] = dLdZ[n][c][counter]# TODO
                        counter += 1 
        dLdA = Z

        return Z

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        k1 = A.shape[2]*self.upsampling_factor - (self.upsampling_factor-1)
        k2 = A.shape[3]*self.upsampling_factor - (self.upsampling_factor-1)
        Z = np.zeros((A.shape[0],A.shape[1],k1,k2))
        Z[:,:,::(self.upsampling_factor),::(self.upsampling_factor)] = A
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA =  dLdZ[:,:,::(self.upsampling_factor),::(self.upsampling_factor)]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        Z = A[:,:,::(self.downsampling_factor),::(self.downsampling_factor)]
        self.shape = A.shape
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        d = np.zeros((self.shape))
        d[:,:,::(self.downsampling_factor),::(self.downsampling_factor)] = dLdZ[:,:,:,:]
        dLdA = d #TODO

        return dLdA