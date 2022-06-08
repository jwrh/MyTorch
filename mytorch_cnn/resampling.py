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
        # n = A.shape[2]
        # a = A.shape[0]
        # b = A.shape[1]
        # nf = n*self.upsampling_factor - (self.upsampling_factor-1)
        # Z = np.zeros((a,b,nf))
        # for n in range(a):
        #     for c in range(b):
        #         counter = 0
        #         for x in range(nf):
        #             if x%self.upsampling_factor == 0:
        #                 Z[n][c][x] = A[n][c][counter]# TODO
        #                 counter += 1 
        # shape of output, formula given, checked
        # assign to zeros
        # use like o[:,:,2]
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # counter = 0
        dLdA =  dLdZ[:,:,::(self.upsampling_factor)]
        # x = np.zeros((dLdZ.shape[0],dLdZ.shape[1],(dLdZ.shape[2]+(self.upsampling_factor-1))//(self.upsampling_factor)))
        # for a in range(x.shape[0]):
        #     for b in range(x.shape[1]):
        #         counter  = 0
        #         for i in range(dLdZ.shape[2]):
        #             if i%self.upsampling_factor == 0:
        #                 # print(dLdZ.shape[2],x.shape[2],counter,i)
        #                 x[a][b][counter] = dLdZ[a][b][i] 
        #                 counter += 1
        # dLdA =  x  #TODO

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
        # print(n,self.downsampling_factor,n+1//self.downsampling_factor)
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
        dLdA =  Z  #TODO
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
        # print("ss shape",n)
        nf = self.boolean
        Z = np.zeros((a,b,nf))
        for n in range(a):
            for c in range(b):
                counter = 0
                for x in range(nf):
                    if x%self.downsampling_factor == 0:
                        Z[n][c][x] = dLdZ[n][c][counter]# TODO
                        counter += 1 
        dLdA = Z  #TODO

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
        # n1 = A.shape[2]
        # n2 = A.shape[3]
        # k1 = A.shape[2]*self.upsampling_factor - (self.upsampling_factor-1)
        # k2 = A.shape[3]*self.upsampling_factor - (self.upsampling_factor-1)
        # Z = np.full((A.shape[0],A.shape[1],k1,k2),0)
        # print(A)
        # for a in range(A.shape[0]):
        #     for b in range(A.shape[1]):
        #         count1 = 0 
        #         for c in range(k1):
        #             count2 = 0
        #             for d in range(k2):
        #                 if ((c%self.upsampling_factor ==0 )and (d%self.upsampling_factor == 0)):
        #                     Z[a, b, c, d] = A[a, b, count1, count2]
        #                     # print(A[a,b,count1,count2])
        #                     count2 += 1 
                            
        #             if c%self.upsampling_factor ==0 :
                        
        #                 count1 += 1
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # print("asad",dLdZ)
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
         # TODO
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
        # print(d)
        d[:,:,::(self.downsampling_factor),::(self.downsampling_factor)] = dLdZ[:,:,:,:]
        # print(d)
        dLdA = d #TODO

        return dLdA