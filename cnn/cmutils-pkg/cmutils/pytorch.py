"""
cmutils/pytorch.py
Utility functions for training a pytorch network on datasets

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""

import os
from pathlib    import Path

import rasterio as rio
import numpy    as np

import warnings

import torch

warnings.filterwarnings("ignore",category=rio.errors.NotGeoreferencedWarning)

def _lab_loader(path):
    lab = rio.open(path).read()
    if lab.shape[0]==3:
        lab = (lab[0]==255) & (lab.sum(axis=0)==255)
    else:
        lab = lab[0]==1
    return lab

#################
# PREPROCESSORS #
#################

class ClampMethaneTile(object):
    """ Clamp matched filter tiles to a value range for neural network stability

    Parameters
    ----------
    ch4min: int
        Minimum clip for the methane layer. Defaults to 250.
        Applied to channel 1 if single-channel, 4 if quad-channel.
    ch4max: int
        Maximum clip for the methane layer. Defaults to 4000.
        Applied to channel 1 if single-channel, 4 if quad-channel.
    rgbmin: int
        Minimum clip for the RGB layers. Defaults to 0.
        Not applied if single-channel, channels 1, 2, 3 if quad-channel.
    rgbmax: int
        Maximum clip for the RGB layers. Defaults to 20.
        Not applied if single-channel, channels 1, 2, 3 if quad-channel.
    
    Returns
    -------
    Preprocessed/clipped tensor array
    """
    def __init__(self, ch4min=0, ch4max=4000, rgbmin=0, rgbmax=20):
        assert ch4max > ch4min
        assert rgbmax > rgbmin

        self.ch4min = ch4min
        self.ch4max = ch4max
        self.rgbmin = rgbmin
        self.rgbmax = rgbmax

    def __call__(self, img):
        # We only expect 1 or 4 channels (CH4 or RGB+CH4)
        assert img.shape[0] == 1 or img.shape[0] == 4

        # Clip values accordingly
        if img.shape[0] == 1:
            return torch.clamp(img, self.ch4min, self.ch4max)
        elif img.shape[0] == 4:
            img[:3] = torch.clamp(img[:3], self.rgbmin, self.rgbmax)
            img[3] = torch.clamp(img[3], self.ch4min, self.ch4max)
            return img

    def __repr__(self):
        return self.__class__.__name__

class ClampScaleMethaneTile(object):
    """ Clamp and scale matched filter tiles to a value range for neural network stability

    Parameters
    ----------
    ch4min: int
        Minimum clip for the methane layer. Defaults to 250.
        Applied to channel 1 if single-channel, 4 if quad-channel.
    ch4max: int
        Maximum clip for the methane layer. Defaults to 4000.
        Applied to channel 1 if single-channel, 4 if quad-channel.
    rgbmin: int
        Minimum clip for the RGB layers. Defaults to 0.
        Not applied if single-channel, channels 1, 2, 3 if quad-channel.
    rgbmax: int
        Maximum clip for the RGB layers. Defaults to 20.
        Not applied if single-channel, channels 1, 2, 3 if quad-channel.
    
    Returns
    -------
    Preprocessed/clipped tensor array
    """
    def __init__(self, ch4min=0, ch4max=4000, rgbmin=0, rgbmax=20):
        assert ch4max > ch4min
        assert rgbmax > rgbmin

        self.ch4min = ch4min
        self.ch4max = ch4max
        self.rgbmin = rgbmin
        self.rgbmax = rgbmax

    def __call__(self, img):
        # We only expect 1 or 4 channels (CH4 or RGB+CH4)
        assert img.shape[0] == 1 or img.shape[0] == 4

        # Clip values accordingly
        if img.shape[0] == 1:
            return torch.clamp(img, self.ch4min, self.ch4max) / (self.ch4max - self.ch4min)
        elif img.shape[0] == 4:
            img[:3] = torch.clamp(img[:3], self.rgbmin, self.rgbmax) / (self.rgbmax - self.rgbmin)
            img[3] = torch.clamp(img[3], self.ch4min, self.ch4max) / (self.ch4max - self.ch4min)
            return img

    def __repr__(self):
        return self.__class__.__name__

class ClampAuxTile(object):
    """ Clamp matched filter tiles to a value range for neural network stability
    Only clamps channel 1

    Parameters
    ----------
    ch4min: int
        Minimum clip for the methane layer. Defaults to 0.
        Applied to channel 1.
    ch4max: int
        Maximum clip for the methane layer. Defaults to 4000.
        Applied to channel 1.
    
    Returns
    -------
    Preprocessed/clipped tensor array
    """
    def __init__(self, auxmin, auxmax):
        self.auxmin = auxmin
        self.auxmax = auxmax

    def __call__(self, img):
        img = torch.clamp(img, self.auxmin, self.auxmax)
        return img

    def __repr__(self):
        return self.__class__.__name__

class ClampAllAuxTile(object):
    """ Clamp matched filter tiles to a value range for neural network stability
    Only clamps channel 1

    Parameters
    ----------
    ch4min: int
        Minimum clip for the methane layer. Defaults to 0.
        Applied to channel 1.
    ch4max: int
        Maximum clip for the methane layer. Defaults to 4000.
        Applied to channel 1.
    
    Returns
    -------
    Preprocessed/clipped tensor array
    """
    def __init__(self, ch4min=0, ch4max=4000, h2omin=0, h2omax=0.6, rgbmin=0, rgbmax=14, 
        ndwimin=0, ndwimax=0.97, ndvimin=0, ndvimax=1.1, endvimin=0, endvimax=0.8, swalbmin=0, swalbmax=1.5):
        assert ch4max > ch4min

        self.ch4min = ch4min
        self.ch4max = ch4max
        self.h2omin = h2omin
        self.h2omax = h2omax
        self.rgbmin = rgbmin
        self.rgbmax = rgbmax
        self.ndwimin = ndwimin
        self.ndwimax = ndwimax
        self.ndvimin = ndvimin
        self.ndvimax = ndvimax
        self.endvimin = endvimin
        self.endvimax = endvimax
        self.swalbmin = swalbmin
        self.swalbmax = swalbmax

    def __call__(self, img):
        img[0] = torch.clamp(img[0], self.ch4min, self.ch4max)
        img[0] /= self.ch4max
        img[1] = torch.clamp(img[1], self.h2omin, self.h2omax)
        img[1] /= self.h2omax
        img[2] = torch.clamp(img[2], self.rgbmin, self.rgbmax)
        img[2] /= self.rgbmax
        img[3] = torch.clamp(img[3], self.ndwimin, self.ndwimax)
        img[3] /= self.ndwimax
        img[4] = torch.clamp(img[4], self.ndvimin, self.ndvimax)
        img[4] /= self.ndvimax
        img[5] = torch.clamp(img[5], self.endvimin, self.endvimax)
        img[5] /= self.endvimax
        img[6] = torch.clamp(img[6], self.swalbmin, self.swalbmax)
        img[6] /= self.swalbmax
        return img

    def __repr__(self):
        return self.__class__.__name__

###########################
# CLASSIFICATION DATASETS #
###########################

class ClassifyDatasetCH4(torch.utils.data.Dataset):
    """ Classification dataset only using the methane channel

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing and augmentation
    
    __getitem__
    -----------
    dict:
        xpath: path of input tile
        x: input data
        y: class 1/0
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Append assumed relative path to dataroot.
        # If CSV paths are absolute, dataroot should be '/'.
        x_path = os.path.join(self.dataroot, x_path)

        # 1 is plume, 0 is not plume, -1 is false positive
        # Convert to binary classification
        # (1,0,-1) => (1), (0, -1)
        y = 1 if int(self.datacsv[idx][1]) == 1 else 0

        # Open tile and add dimension for batching
        # Last band is always CH4
        ds = rio.open(x_path)
        x = ds.read(ds.count)
        x = np.expand_dims(x, axis=0)

        # Apply data transformations
        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)

        # Return input, label pair
        return {
            "xpath": x_path,
            "x": x,
            "y": y
        }

class ClassifyDatasetRGBCH4(torch.utils.data.Dataset):
    """ Classification dataset using RGB and methane channels

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing and augmentation
    
    __getitem__
    -----------
    dict:
        xpath: path of input tile
        x: input data
        y: class 1/0
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Append assumed relative path to dataroot.
        # If CSV paths are absolute, dataroot should be '/'.
        x_path = os.path.join(self.dataroot, x_path)

        # 1 is plume, 0 is not plume, -1 is false positive
        # Convert to binary classification
        # (1,0,-1) => (1), (0, -1)
        y = 1 if int(self.datacsv[idx][1]) == 1 else 0

        # Open tile
        x = rio.open(x_path).read()

        # Apply data transformations
        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)

        # Return input, label pair
        return {
            "xpath": x_path,
            "x": x,
            "y": y
        }

class ClassifyDatasetCH4AUX(torch.utils.data.Dataset):
    """ Classification dataset using methane and auxiliary channels

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing and augmentation
    
    __getitem__
    -----------
    dict:
        xpath: path of input tile
        x: input data
        y: class 1/0
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None, channels=None):
        self.channels = channels
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Append assumed relative path to dataroot.
        # If CSV paths are absolute, dataroot should be '/'.
        x_path = os.path.join(self.dataroot, x_path)

        # 1 is plume, 0 is not plume, -1 is false positive
        # Convert to binary classification
        # (1,0,-1) => (1), (0, -1)
        y = 1 if int(self.datacsv[idx][1]) == 1 else 0

        # Open tile
        if self.channels is None:
            x = rio.open(x_path).read()
        else:
            x = rio.open(x_path).read(self.channels)

        # Apply data transformations
        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)

        # Return input, label pair
        return {
            "xpath": x_path,
            "x": x,
            "y": y
        }

class ClassifyStratifiedMultiDatasetCH4(torch.utils.data.Dataset):
    """ Stratified Multicampaign classification dataset only using the methane
    channel. The strategy is to repeat campaigns with fewer tiles to be
    equivalently weighted to a campaign with more tiles.

    Parameters
    ----------
    dataroots: list of str
        Directory paths to campaign datasets
    datacsv: list of list
        List of tile/label pairs from each campaign
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing and augmentation
    
    __getitem__
    -----------
    dict:
        xpath: path of input tile
        x: input data
        y: class 1/0
    """

    def __init__(self, dataroots, datacsvs, transform):
        # Get lengths of each dataset
        c_lens = [len(d) for d in datacsvs]
        # Calculate how many times each smaller dataset has
        strat_ratios = [max(c_lens) // l for l in c_lens]

        self.paths = []
        self.labels = []
        for dataroot, datacsv, sr in zip(dataroots, datacsvs, strat_ratios):
            # For each dataset, repeat the dataset as necessary
            # Note that we guarantee tiles will be repeated the same amount
            for i in range(sr):
                for dc in datacsv:
                    self.paths.append(os.path.join(dataroot, dc[0]))
                    self.labels.append(int(int(dc[1]) == 1))

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 1 is plume, 0 is not plume, -1 is false positive
        # Convert to binary classification
        # (1,0,-1) => (1), (0, -1)
        y = 1 if self.labels[idx] == 1 else 0

        # Open tile and add dimension for batching
        # Last band is always CH4
        ds = rio.open(self.paths[idx])
        x = ds.read(ds.count)
        x = np.expand_dims(x, axis=0)
        # ch x h x w

        # Apply data transformations
        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)

        # Return input, label pair
        return {
            "xpath": self.paths[idx],
            "x": x,
            "y": y
        }

#########################
# SEGMENTATION DATASETS #
#########################

class SegmentDatasetCH4(torch.utils.data.Dataset):
    """ Segmentation dataset only using the methane channel

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing, only applied to X
    augment: obj
        torchvision transform object for augmentation, applied to X and Y

    __getitem__
    -----------
    dict:
        xpath: path of input tile
        ypath: path of label tile
        x: input data
        y: label mask
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None, augment=None):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Class for label generation
        y_class = int(self.datacsv[idx][1])

        # Append assumed relative path to dataroot.
        # If CSV paths are absolute, dataroot should be '/'.
        x_path = os.path.join(self.dataroot, x_path)
        y_path = str(Path(x_path).with_suffix(".png"))

        # Open tile and add dimension for batching
        # Last band is always CH4
        ds = rio.open(x_path)
        x = ds.read(ds.count)
        x = np.expand_dims(x, axis=0)
        # (ch, h, w)

        y = np.array(_lab_loader(y_path))
        x = torch.as_tensor(x, dtype=torch.float)

        # Apply preprocessing
        if self.transform is not None:
            x = self.transform(x)
            
        if y_class != 1:
            # Some tiles that do not contain a plume still have valid masks.
            y = torch.zeros(y.shape)
        else:
            # Only 1 are actual plumes - other values are not.
            y[y!=1] = 0

        y = np.expand_dims(y, axis=0)    
        y = torch.as_tensor(y, dtype=torch.float)            

        # Apply augmentation
        # Need to stack so flip,rot,crop gets applied equally
        # NOTE: Potentially risky if augment introduces aliasing
        if self.augment is not None:
            z = torch.cat((x,y), axis=0)
            z = self.augment(z)
            x = torch.unsqueeze(z[0], dim=0)
            y = torch.unsqueeze(z[1], dim=0)
                
        # Return input, label pair
        return {
            "xpath": x_path,
            "ypath": y_path,
            "x": x,
            "y": y
        }

class SegmentDatasetCH4OneHot(torch.utils.data.Dataset):
    """ Segmentation dataset only using the methane channel

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing, only applied to X
    augment: obj
        torchvision transform object for augmentation, applied to X and Y

    __getitem__
    -----------
    dict:
        xpath: path of input tile
        ypath: path of label tile
        x: input data
        y: label mask
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None, augment=None):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform
        self.augment = augment
        self.onehot = True
        
    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Class for label generation
        y_class = int(self.datacsv[idx][1])

        # Append assumed relative path to dataroot.
        # If CSV paths are absolute, dataroot should be '/'.
        x_path = os.path.join(self.dataroot, x_path)
        y_path = str(Path(x_path).with_suffix(".png"))

        # Open tile and add dimension for batching
        # Last band is always CH4
        ds = rio.open(x_path)
        x = ds.read(ds.count)
        x = np.expand_dims(x, axis=0)
        # (ch, h, w)

        y = np.array(_lab_loader(y_path))
        x = torch.as_tensor(x, dtype=torch.float)

        # Apply preprocessing
        if self.transform is not None:
            x = self.transform(x)
            
        if y_class != 1:
            # Some tiles that do not contain a plume still have valid masks.
            y = torch.zeros(y.shape)
        else:
            # Only 1 are actual plumes - other values are not.
            y[y!=1] = 0

        if self.onehot:
            y = np.stack([1-y,y], axis=0)
        else:
            y = np.expand_dims(y, axis=0)
            
        y = torch.as_tensor(y, dtype=torch.float)            

        # Apply augmentation
        # Need to stack so flip,rot,crop gets applied equally
        # NOTE: Potentially risky if augment introduces aliasing
        if self.augment is not None:
            z = torch.cat((x,y), axis=0)
            z = self.augment(z)
            x = torch.unsqueeze(z[0], dim=0)
            if self.onehot:
                y = z[1:]
            else:
                y = torch.unsqueeze(z[1], dim=0)
                
        # Return input, label pair
        return {
            "xpath": x_path,
            "ypath": y_path,
            "x": x,
            "y": y
        }    

class SegmentClassifyDatasetCH4(torch.utils.data.Dataset):
    """ Segmentation and classification dataset only using the methane channel

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label]
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing, only applied to X
    augment: obj
        torchvision transform object for augmentation, applied to X and Y

    __getitem__
    -----------
    dict:
        xpath: path of input tile
        ypath: path of label tile
        x: input data
        y: label mask
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None, augment=None):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Class for label generation
        y_class = 1 if int(self.datacsv[idx][1]) == 1 else 0

        # Append assumed relative path to dataroot.
        # If CSV paths are absolute, dataroot should be '/'.
        x_path = os.path.join(self.dataroot, x_path)
        y_path = str(Path(x_path).with_suffix(".png"))

        # Open tile and add dimension for batching
        # Last band is always CH4
        ds = rio.open(x_path)
        x = ds.read(ds.count)
        x = np.expand_dims(x, axis=0)
        # (ch, h, w)

        y = np.array(_lab_loader(y_path))
        y = np.expand_dims(y, axis=0)

        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float)

        # Apply preprocessing
        if self.transform is not None:
            x = self.transform(x)
        if y_class != 1:
            # Some tiles that do not contain a plume still have valid masks.
            y = torch.zeros(y.shape)
        else:
            # Only 1 are actual plumes - other values are not.
            y[y!=1] = 0

        # Apply augmentation
        # Need to stack so flip,rot,crop gets applied equally
        # NOTE: Potentially risky if augment introduces aliasing
        if self.augment is not None:
            z = torch.cat((x,y), axis=0)
            z = self.augment(z)
            x = torch.unsqueeze(z[0], dim=0)
            y = torch.unsqueeze(z[1], dim=0)

        # Return input, label pair
        return {
            "xpath": x_path,
            "ypath": y_path,
            "x": x,
            "y": y,
            "class": y_class
        }

class SegmentDatasetRGBCH4(torch.utils.data.Dataset):
    """ Segmentation dataset only using the methane channel

    Parameters
    ----------
    dataroot: str
        Directory path to campaign dataset
    datacsv: list
        List of [relative tile path, label] 
        tilepath is relative to dataroot, label is -1, 0, or 1
    transform: obj
        torchvision transform object for preprocessing, only applied to X
    augment: obj
        torchvision transform object for augmentation, applied to X and Y

    __getitem__
    -----------
    dict:
        xpath: path of input tile
        ypath: path of label tile
        x: input data
        y: label mask
    """

    def __init__(self, dataroot='/', datacsv=None, transform=None, augment=None):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Path to image
        x_path = self.datacsv[idx][0]

        # Class for label generation
        y_class = int(self.datacsv[idx][1])

        # Some older datasets have absolute paths instead of relative paths
        # Corrects these absolute paths into relative paths; no effect otherwise
        x_path = os.path.join(self.dataroot, x_path)
        y_path = str(Path(x_path).with_suffix(".png"))

        # Open tile and add dimension for batching
        x = rio.open(x_path).read()
        # (ch, h, w)

        y = np.array(_lab_loader(y_path))
        y = np.expand_dims(y, axis=0)
        # (ch, h, w)

        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float)

        # Apply preprocessing
        if self.transform is not None:
            x = self.transform(x)
        if y_class != 1:
            # Some tiles that do not contain a plume still have valid masks.
            y = torch.zeros(y.shape)
        else:
            # Only ones are actual plumes - other values are not.
            y[y!=1] = 0

        # Apply augmentation
        # Need to stack so flip,rot,crop gets applied equally
        # NOTE: Potentially risky if augment introduces aliasing
        if self.augment is not None:
            z = torch.cat((x,y), axis=0)
            z = self.augment(z)
            x = z[:-1]
            y = z[-1]
            y = torch.unsqueeze(y, dim=0)

        # Return input, label pair
        return {
            "xpath": x_path,
            "ypath": y_path,
            "x": x,
            "y": y
        }
