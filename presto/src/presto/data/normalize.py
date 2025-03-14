#!/usr/bin/env python3

import torch as th
import torch.nn as nn


class Normalize(nn.Module):
    """
    Utility class to normalize tensors.
    Various options are supported for instancing.
    @see from_{minmax,avgstd,data} variants.
    """

    def __init__(self,
                 center: th.Tensor,
                 radius: th.Tensor):
        super().__init__()
        self.register_buffer('center',
                             center, persistent=True)
        self.register_buffer('radius',
                             radius, persistent=True)

    def normalize(self, x: th.Tensor,
                  inplace: bool = False):
        """
        y = (x - c) / s
        """
        if not inplace:
            x = x.clone()
        # return x.sub_(self.center).div_(self.radius)
        return (x
                .reshape(-1, x.shape[-1])
                .sub_(self.center[None].to(device=x.device))
                .div_(self.radius[None].to(device=x.device))
                .reshape(x.shape))

    def unnormalize(self, x: th.Tensor,
                    inplace: bool = False):
        """
        x = y*s + c
        """
        if not inplace:
            x = x.clone()
        return (x
                .reshape(-1, x.shape[-1])
                .mul_(self.radius[None].to(device=x.device))
                .add_(self.center[None].to(device=x.device))
                .reshape(x.shape))

    def __call__(self, x: th.Tensor,
                 inplace: bool = False):
        return self.normalize(x, inplace)

    @classmethod
    def from_minmax(cls, xmin, xmax):
        """ map to (-1, +1) """
        xavg = 0.5 * (xmin + xmax)
        xstd = 0.5 * (xmax - xmin)
        return cls(xavg, xstd)

    @classmethod
    def from_avgstd(cls, xavg, xstd):
        """ map to unit-normal distribution """
        return cls(xavg, xstd)

    @classmethod
    def from_data(cls,
                  x: th.Tensor,
                  norm_type: str = 'minmax',
                  dim: int = 0):
        if norm_type == 'minmax':
            xmin = x.min(dim=dim)
            xmax = x.max(dim=dim)
            return cls.from_minmax(xmin, xmax)
        elif norm_type == 'avgstd':
            xavg = x.mean(dim=dim)
            xstd = x.std(dim=dim)
            return cls.from_avgstd(xavg, xstd)
        else:
            raise ValueError(F'Unknown norm_type={norm_type}')

    @classmethod
    def identity(cls, *args, **kwds):
        """ Load parameters to skip normalization """
        return cls(th.as_tensor(0.0, *args, **kwds),
                   th.as_tensor(1.0, *args, **kwds))
