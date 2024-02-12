"""
cmutils/fcn.py
Utility functions for converting CNN to FCN

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""

import numpy as np
import torch
from torch import nn

class ShiftStitchDataset(torch.utils.data.Dataset):
    """ Single flightline for shift and stitching
    
    Usage:
    ShiftStitchDataset(flightlinepath, transform, scale)
    """
    
    def __init__(self, x, transform, scale=32):
        self.x = x
        self.x_shape = x.shape
        self.transform = transform
        self.scale = scale

        pad0 = scale - (self.x_shape[0] % self.scale)
        pad1 = scale - (self.x_shape[1] % self.scale)

        # Left Right Top Bottom
        self.div_pad = nn.ZeroPad2d((0, pad1, 0, pad0))

    def __len__(self):
        return self.scale ** 2
    
    def __getitem__(self, idx):
        # Calculate shift-and-stitch padding for this index
        top = idx // self.scale
        left = idx % self.scale
        
        t = torch.as_tensor(self.x, dtype=torch.float).unsqueeze(0)
        if self.transform is not None:
            t = self.transform(t)

        # Divisibility padding
        t = self.div_pad(t)

        # Shift-and-Stitch padding
        # Left Right Top Bottom
        t = nn.ZeroPad2d((left, self.scale-left, top, self.scale-top))(t)
        return (top, left), t 

def stitch_stack(fl_shape, ts, ls, predstack, scale=32):
    """ Interlace shifted outputs

    fl_shape: Shape of original flightline for cropping
    ts: List of top shifts
    ls: List of left shifts
    predstack: Stack of shifted predictions
    scale: Downscale factor of model, default 32.
    """
    # Storage for final stitched output
    stitched = np.zeros(shape=(predstack.shape[1]*scale, predstack.shape[2]*scale))

    # Iterate through shifts and outputs
    for i in range(predstack.shape[0]):
        top = ts[i]
        left = ls[i]
        # Save them to corresponding strided pixels
        stitched[scale-top-1::scale, scale-left-1::scale] = predstack[i]

    # Crop the top left
    #stitched = stitched[:fl_shape[0], :fl_shape[1]]
    # Crop the center
    #stitched = stitched[scale//2:fl_shape[0]+scale//2, scale//2:fl_shape[1]+scale//2]
    # Crop the bottom right
    stitched = stitched[scale:fl_shape[0]+scale, scale:fl_shape[1]+scale]

    return stitched


def cnn_to_fcn(model, truncate=-5, pool=(8,8), in_ch=1024):
    """
    Convert a classification CNN into a saliency map FCN
    Assumes that the final classification layer can be referenced via model.fc

    Parameters
    ----------
    model: pytorch model
    truncate: int
        All layers prior to this index will be cut off. Should refer to the
        final global pooling layer.
    pool: tuple
        Size of the appended stride-1 pooling layer. Should be the effective
        size of the final global pooling kernel when the model was trained.
        For example, a model trained on 256x256 tiles with a 32x downsampling
        factor will have a pool kernel size (8,8)
    in_ch: int
        Number of expected channels at the truncated layer
    """
    # Cut off global pooling, FC layers, and any auxiliary layers
    fcn = nn.Sequential(*list(model.children())[:truncate])
    
    # Add a stride-1 average pooling layer
    fcn.add_module('pool_repl', nn.AvgPool2d(pool, 1, padding=pool[0]//2, ceil_mode=False, count_include_pad=False))
    
    if pool[0] % 2 == 0:
        # An even kernel size requires cropping
        fcn.add_module('pool_crop', nn.ConstantPad2d((0, -1, 0, -1), 0))
    
    # Replace fully connected layer with (1,1) convolution
    fcn.add_module('final_conv', nn.Conv2d(in_ch, 2, kernel_size=1))
    
    # Copy weights from fully connected layer to convolutional layer
    fcn.final_conv.weight.data.copy_(model.fc.weight.data[:,:,None,None])
    fcn.final_conv.bias.data.copy_(model.fc.bias.data)

    return fcn