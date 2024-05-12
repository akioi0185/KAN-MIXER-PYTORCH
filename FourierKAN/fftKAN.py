import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class NaiveFourierKANLayer(torch.nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize = 10,addbias=True):
        super(NaiveFourierKANLayer,self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
         # Learnable gridsize parameter
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))

        # Fourier coefficients as a learnable parameter with Xavier initialization
        self.fouriercoeffs = nn.Parameter(torch.empty(2, outdim, inputdim, initial_gridsize))
        nn.init.xavier_uniform_(self.fouriercoeffs)


        if( self.addbias ):
            self.bias  = torch.nn.Parameter( torch.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()

        original_shape = x.shape
        x = x.contiguous()
        x = x.view(-1, original_shape[-1])
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,) #batch + outputdim

        x = torch.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape( torch.arange(1,gridsize+1,device=x.device),(1,1,1,gridsize))

        #k shape: 1, 1, 1, gridesize

        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) )
        #xrshp shape: batch, 1, inputdim, 1

        #This should be fused to avoid materializing memory
        c = torch.cos( k*xrshp )
        s = torch.sin( k*xrshp )

        #c, s shape: batch 1 inputsize gridesize

        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1))
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))

        if( self.addbias):
            y += self.bias
        #End fuse
        #y = torch.reshape( y, outshape)

        y = y.view(*original_shape[:-1],-1)
        return y