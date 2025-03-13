#!/usr/bin/env python3

import torch as th
import numpy as np


def randu(size, lo, hi, *args, **kwds) -> th.Tensor:
    """
    Generate a random uniform tensor of size `size`
    between `lo` and `hi`.
    """
    z = th.rand(size, *args, **kwds)
    kwds.pop('generator', None)
    lo = th.as_tensor(lo, *args, **kwds)
    hi = th.as_tensor(hi, *args, **kwds)
    return z * (hi - lo) + lo


def dcn(x: th.Tensor) -> np.ndarray:
    """
    Convert torch tensor into numpy array.
    """
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return x
