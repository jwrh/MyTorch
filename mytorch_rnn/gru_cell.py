import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.z5 = np.dot(self.Wrx, x)
        self.z6 = np.dot(self.Wrh, h)
        self.z7 = self.z5 + self.z6 + self.brx + self.brh
        self.z8 = self.r_act(self.z7)
        self.r = self.z8

        self.z1 = np.dot(self.Wzh, h)
        self.z2 = np.dot(self.Wzx, x)
        self.z3 = self.z1 + self.z2 + self.bzh+ self.bzx
        self.z4 = self.z_act(self.z3)
        self.z = self.z4

        

        self.z9 = np.dot(self.Wnh, h) + self.bnh
        self.z10 = self.z8 * self.z9
        self.z11 = np.dot(self.Wnx, x)
        self.z12 = self.z10 + self.z11 + self.bnx
        self.z13 = self.h_act(self.z12)
        self.n = self.z13

        self.z14 = 1 - self.z4
        self.z15 = self.z14 * self.z13
        self.z16 = self.z4 * h
        self.z17 = self.z15 + self.z16
        h_t = self.z17
        
        # This code should not take more than 10 lines. 
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        # raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        dh = delta * self.z4
        self.dz_t = delta * self.hidden
        self.dz_t -= delta * self.z13
        self.dn_t = delta * (1- self.z4)
        self.inside = self.dn_t  * (1 - self.h_act(self.z12) * self.h_act(self.z12)).T




        self.dWnx += np.dot(self.inside.T, self.x.reshape(1,-1))
        self.dbnx += np.mean(self.inside,axis=0)
        self.dr_t = self.inside * self.z9
        dx = np.dot(self.inside,self.Wnx)
        self.kinside = self.inside * self.r
        self.dbnh += np.mean(self.kinside,axis=0)
        self.dWnh += np.dot(self.kinside.T, self.hidden.reshape(1,-1))
        dh += np.dot(self.kinside, self.Wnh)
        self.zinside = self.dz_t * self.r_act(self.z3) * (1 - self.r_act(self.z3))
        self.dbzx += np.mean(self.zinside,axis=0)
        self.dbzh += np.mean(self.zinside,axis=0)
        self.dWzx += np.dot(self.zinside.T, np.reshape(self.x, (1, -1)))
        dx += np.dot(self.zinside, self.Wzx)
        self.dWzh += np.dot(self.zinside.T, np.reshape(self.hidden, (1, -1)))
        dh += np.dot(self.zinside, self.Wzh)
        self.rinside = self.dr_t * self.r_act(self.z7) * (1 - self.r_act(self.z7))
        self.dbrx += np.mean(self.rinside,axis=0)
        self.dbrh += np.mean(self.rinside,axis=0)
        self.dWrx += np.dot(self.rinside.T, np.reshape(self.x, (1, -1)))
        dx += np.dot(self.rinside, self.Wrx)
        self.dWrh += np.dot(self.rinside.T, np.reshape(self.hidden, (1, -1)))
        dh += np.dot(self.rinside, self.Wrh)
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        
