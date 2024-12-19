import torch
import numpy as np
import torch.func as func
from functorch import make_fx
from typing import Callable, Optional, Tuple

class Subbox :
    """
    A class to handle sub-box operations for map2map forward modeling.

    Attributes
    ----------
    model : callable
        The model function to evaluate.
    jvp_dir : torch.Tensor
        The direction vector for the Jacobian-vector product derivative.
    size : tuple
        The size of the box (number of grid sites along each dimension).
    ndiv : tuple
        The number of subbox divisions in each dimension.
    device : torch.device
        The device to perform computations on.
    pad : tuple
        The padding to be applied to each side of each dimension.
    crop_inds : tuple or None
        The indices for cropping the sub-box from the full box.
    dis_out : torch.Tensor
        The output tensor for displacement.
    vel_out : torch.Tensor
        The output tensor for velocity. 
    
    Methods
    -------
    setCrop(indx)
        Sets the crop indices for the sub-box.
    getCropInds(anchor, crop, pad, size)
        Computes the crop indices for the sub-box.
    forward(input, Om, Dz, indx=None)
        Performs the forward pass of the model on the sub-box.
    """

    NDIM = 3  # Number of dimensions

    def __init__(self, model: Callable, in_chan: int, size: Tuple[int, int, int], ndiv: Tuple[int, int, int], 
                 padding: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], device: str, indx: Optional[int] = None):
        """
        Initializes the Subbox instance.

        Parameters
        ----------
        model : callable
            The model function to evaluate.
        in_chan : int
            The number of the input channels.
        size : tuple
            The size of the box (number of grid sites along each dimension).
        ndiv : tuple
            The number of subbox divisions in each dimension.
        padding : tuple
            The padding to be applied to each side of each dimension.
        device : torch.device
            The device to perform computations on.
        indx : int, optional
            The index of the subbox (default is None).
        """
        self.model = model
        self.device = device
        self.first_round = True
        self.first_jvp_round = True
        
        self.size = size
        self.ndiv = ndiv
        self.pad = padding
        
        spatial_in_shape = (1, in_chan,) + tuple(s // d + np.sum(p) for d, s, p in zip(ndiv, size, padding))
        #self.jvp_dir = torch.tensor([0, 1.]).float().to(device)
        self.jvp_dir = torch.tensor([1.], device=device)        
        #self.jvp_dir = (torch.zeros(spatial_in_shape).float().to(device), torch.tensor([0, 1.]).float().to(device))    
        
        self.crop_inds = None
        if indx is not None:
            self.setCrop(indx)
                    
        self.dis_out = torch.zeros(self.NDIM, *size, device='cpu', dtype=torch.float32)
        self.vel_out = torch.zeros(self.NDIM, *size, device='cpu', dtype=torch.float32)

    def setCrop(self, indx: int) -> None:
        """
        Sets the crop indices for the subbox.

        Parameters
        ----------
        indx : int
            The index of the subbox.
        """
        crop = tuple(s // d for s, d in zip(self.size, self.ndiv))
        anchor = (
            (indx // self.ndiv[1] // self.ndiv[2]) * (self.size[0] // self.ndiv[0]),
            ((indx // self.ndiv[1]) % self.ndiv[2]) * (self.size[1] // self.ndiv[1]),
            ((indx % self.ndiv[1]) % self.ndiv[2]) * self.size[2] // self.ndiv[2]
        )
        self.crop_inds = self.getCropInds(anchor, crop, self.pad, self.size)
        self.add_inds = self.getCropInds(anchor, crop, ((0,)*2,)*self.NDIM, self.size)

    def getCropInds(self, anchor: Tuple[int, int, int], crop: Tuple[int, int, int], 
                    pad: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], 
                    size: Tuple[int, int, int]) -> Tuple[slice, ...]:
        """
        Computes the crop indices for the subbox.

        Parameters
        ----------
        anchor : tuple
            The anchor point for the crop.
        crop : tuple
            The size of the crop.
        pad : tuple
            The padding to be applied to each dimension.
        size : tuple
            The size of the full box.

        Returns
        -------
        Tuple[slice, ...]
            The computed crop indices.
        """        
        ind = [slice(None)]
        for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, crop, pad, size)):
            start = a - p0
            end = a + c + p1
            i = np.arange(start, end) % s
            ind.append(i.reshape((-1,) + (1,) * (self.NDIM - d - 1)))        
        return tuple(ind)

    def forward(self, input: torch.Tensor, Om: torch.Tensor, Dz: torch.Tensor, indx: Optional[int] = None, compile: bool = False) -> None :
        """
        Performs the forward pass of the model on the subbox and adds the result to dis_out.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor.
        Om : torch.Tensor
            The matter fraction parameter tensor.
        Dz : torch.Tensor
            The growth factor parameter tensor.
        indx : int, optional
            The index of the subbox (default is None).
        """
        if indx is not None:
            self.setCrop(indx)
                        
        if self.crop_inds is None:
            raise ValueError('A subbox crop has not been set yet')
        
        dis_in = input[self.crop_inds].unsqueeze(0).to(self.device)


        def eval_model(Dz) :
            return self.model(dis_in, Om, Dz)
        
        '''
        if self.first_round :
            def eval_model(Dz) :
                return self.model(dis_in, Om, Dz)
            if compile :
                eval_model = make_fx(eval_model)(Dz)
                with torch.jit.optimized_execution(True) :
                    eval_model = torch.jit.trace(eval_model, (Dz))
                    eval_model = torch.jit.script(eval_model)
            self.eval_model = eval_model
            self.first_round = False
        '''
            
        dis_out, _ = eval_model(Dz)

        dis_out = dis_out.detach().squeeze(0).cpu()
                
        self.dis_out[self.add_inds] += dis_out
        
        return
    
    def forward_jvp(self, input: torch.Tensor, Om: torch.Tensor, Dz: torch.Tensor, indx: Optional[int] = None, compile: bool = False) -> None :
        """
        Performs the forward pass of the model and its jvp forward derivative on the subbox and adds the result to dis_out and vel_out.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor.
        Om : torch.Tensor
            The matter fraction parameter tensor.
        Dz : torch.Tensor
            The growth factor parameter tensor.
        indx : int, optional
            The index of the subbox (default is None).
        """
        if indx is not None:
            self.setCrop(indx)
                        
        if self.crop_inds is None:
            raise ValueError('A subbox crop has not been set yet')
        
        dis_in = input[self.crop_inds].unsqueeze(0).to(self.device)
 
        def eval_model(Dz) :
            return self.model(dis_in, Om, Dz)

        '''
        if self.first_jvp_round :
            def eval_model(Dz) :
                return self.model(dis_in, Om, Dz)
            def eval_model_jvp(Dz) :
                return func.jvp(eval_model, (Dz,), (self.jvp_dir,))
            if compile :
                eval_model_jvp = make_fx(eval_model_jvp)(Dz)
                with torch.jit.optimized_execution(True) :
                    eval_model_jvp = torch.jit.trace(eval_model_jvp, (Dz))
                    eval_model_jvp = torch.jit.script(eval_model_jvp)
            self.eval_model_jvp = eval_model_jvp
            self.first_jvp_round = False
        '''
            
        (dis_out, _), (vel_out, _) = func.jvp(eval_model, (Dz,), (self.jvp_dir,))

        dis_out = dis_out.detach().squeeze(0).cpu()
        vel_out = vel_out.detach().squeeze(0).cpu()   
                
        self.dis_out[self.add_inds] += dis_out
        self.vel_out[self.add_inds] += vel_out
        
        return
