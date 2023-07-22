# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        w_0 = A.shape[2]
        w_1 = w_0-self.kernel_size+1 #output size
        N = A.shape[0]
        c_1 = self.W.shape[0]
        Z = np.zeros((N,c_1,w_0))
        ap = np.zeros((N,self.in_channels,self.kernel_size,w_1))
        for n in range(w_1):
          ap[:,:,:,n] = A[:,:,n:n+self.kernel_size]
        Z = np.tensordot(ap,self.W,axes = ([1,2],[1,2]))
        Z = np.transpose(Z,axes = [0,2,1])
        b = np.full((Z.shape),self.b[0])
        Z += b
        
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        w_0 = self.A.shape[2]
        w_1 = w_0-dLdZ.shape[2]+1 
        N = self.A.shape[0]
        ap = np.zeros((N,self.in_channels,dLdZ.shape[2],w_1))
        for n in range(w_1):
          ap[:,:,:,n] = self.A[:,:,n:n+dLdZ.shape[2]]
        self.dLdW = np.tensordot(dLdZ,ap,axes = ([0,2],[0,2]))
        self.dLdW = np.transpose(self.dLdW,axes = [0,1,2])  
        self.dLdb = np.sum(dLdZ,axis =(0,2))
        b  = np.zeros((dLdZ.shape[0],dLdZ.shape[1],dLdZ.shape[2]+2*(self.kernel_size-1)))
        b[:,:,(self.kernel_size-1):(dLdZ.shape[2]+self.kernel_size-1)] = dLdZ[:,:,:]
        k = np.zeros(self.W.shape)
        k =np.flip(self.W,axis = 2)
        al = np.zeros((b.shape[0],b.shape[1],self.kernel_size,self.A.shape[2]))
        for l in range(self.A.shape[2]):
            al[:,:,:,l] = b[:,:,l:l+self.kernel_size]
        dLdA = np.zeros((b.shape[0],self.in_channels,w_0))
        j = np.tensordot(al,k,axes = [(1,2),(0,2)])
        dLdA = np.transpose(j,axes = [0,2,1]))
        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None): 
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 =  Conv1d_stride1(in_channels, out_channels, kernel_size) # TODO
        self.downsample1d = Downsample1d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        Z  = self.conv1d_stride1.forward(A)
        # Call Conv1d_stride1
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        dLdA = self.downsample1d.backward(dLdZ)
        # TODO
        
        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA) # TODO 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        h_1 = A.shape[2]-self.kernel_size+1
        w_1 = A.shape[3]-self.kernel_size+1
        ap = np.zeros((A.shape[0],A.shape[1],self.kernel_size,h_1,A.shape[3]))
        for i in range(h_1):
            ap[:,:,:,i] = A[:,:,i:i+self.kernel_size]
        app = np.zeros((A.shape[0],A.shape[1],self.kernel_size,h_1,self.kernel_size,w_1))
        for k in range(w_1):
            app[:,:,:,:,:,k] = ap[:,:,:,:,k:k+self.kernel_size]
        result = np.tensordot(app,self.W,axes = [(1,2,4),(1,2,3)])
        result = np.transpose(result, axes = [0,3,1,2])


        return result

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        w_0 = self.A.shape[2]
        w_1 = w_0-dLdZ.shape[2]+1 #output size
        h_0 = self.A.shape[3]
        h_1 = h_0-dLdZ.shape[3]+1
        N = self.A.shape[0]
        ap = np.zeros((N,self.in_channels,dLdZ.shape[2],w_1,h_0))
        for n in range(w_1):
          ap[:,:,:,n] = self.A[:,:,n:n+dLdZ.shape[2]]
        app = np.zeros((N,self.in_channels,dLdZ.shape[2],w_1,dLdZ.shape[3],h_1))
        for n in range(h_1):
            app[:,:,:,:,:,n] = ap[:,:,:,:,n:n+dLdZ.shape[2]]
        self.dLdW = np.tensordot(dLdZ,app,axes = ([0,2,3],[0,2,4]))
        self.dLdb = np.sum(dLdZ,axis =(0,2,3))
        b  = np.zeros((dLdZ.shape[0],dLdZ.shape[1],dLdZ.shape[2]+2*(self.kernel_size-1),dLdZ.shape[3]+2*(self.kernel_size-1)))
        b[:,:,(self.kernel_size-1):(dLdZ.shape[2]+self.kernel_size-1),(self.kernel_size-1):(dLdZ.shape[3]+self.kernel_size-1)] = dLdZ[:,:,:,:]
        k = np.zeros(self.W.shape)
        k =np.flip(self.W,axis = (2,3))
        al = np.zeros((b.shape[0],b.shape[1],self.kernel_size,self.A.shape[2],b.shape[3]))
        for l in range(self.A.shape[2]):
            al[:,:,:,l] = b[:,:,l:l+self.kernel_size]
        all = np.zeros((b.shape[0],b.shape[1],self.kernel_size,self.A.shape[2],self.kernel_size,self.A.shape[3]))
        for i in range(self.A.shape[3]):
            all[:,:,:,:,:,i] = al[:,:,:,:,i:i+self.kernel_size]
        dLdA = np.zeros((b.shape[0],self.in_channels,w_0))
        j = np.tensordot(all,k,axes = [(1,2,4),(0,2,3)])
        dLdA = np.transpose(j,axes = [0,3,1,2])
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 =Conv2d_stride1( in_channels, out_channels,
                 kernel_size)
        self.downsample2d =  Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)
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
        dLdA = self.conv2d_stride1.backward(dLdA)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,  weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # upsample
        Q = self.upsample1d.forward(A)

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(Q)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)

        dLdA =  self.upsample1d.backward(delta_out)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,  weight_init_fn, bias_init_fn) 
        self.upsample2d =Upsample2d(upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) 

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)


        dLdA = self.upsample2d.backward(delta_out)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        Z = np.reshape(A,(A.shape[0],A.shape[1]*A.shape[2]))
        self.shape = A.shape
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ,self.shape)

        return dLdA

