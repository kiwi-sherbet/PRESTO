#!/usr/bin/env python3

from typing import List, Dict
from tqdm.auto import tqdm


def rename_obs(cs: List[Dict],
               disable_tqdm: bool = True):
    """
    rename obstacles so that there are no duplicates across batches.
    """
    count: int = 0
    out = []
    for c in tqdm(cs, desc='relabel',
                  disable=disable_tqdm):
        out_i = {}
        for geom_type, geoms in c.items():
            out_i[geom_type] = {}
            for name, geom in geoms.items():
                if geom_type == 'mesh':
                    # keep mesh names
                    out_i[geom_type][name] = geom
                else:
                    if 'base' in name:
                        # keep base names
                        out_i[geom_type][F'base'] = geom
                    else:
                        # rename obstacles
                        out_i[geom_type][F'{count:03d}'] = geom
                        count += 1
        out.append(out_i)
    return out
