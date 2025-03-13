#!/usr/bin/env python3

import re
from typing import Tuple
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pickle
from icecream import ic

from presto.util.torch_util import dcn
from presto.util.path import ensure_directory
from presto.util.config import with_oc_cli_v2


@dataclass
class Config:
    pattern: str = F'run-*-NEW2-*P*-N*S64-*.pkl/*.pkl'
    out_path: str = '/tmp/metrics/'
    runs: Tuple[str, ...] = ()


@with_oc_cli_v2
def main(cfg: Config = Config()):
    pattern = re.compile(
        r'(run-\d+-NEW2-O(\d+)-D(\d)-P(\d)-N(\d+)-obj-(\d)-\d(-v2)?-S64(-G(\d))?(-C(\d))?)'
    )

    metricss = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in tqdm(sorted(Path('.').glob(cfg.pattern)), desc='load'):
        key = p.parent.stem
        if '2-2' in key and '2-2-v2' not in key:
            continue

        found = False
        for run in cfg.runs:
            if run in key:
                found = True
                break
        if not found:
            continue

        match = pattern.match(key)
        if not match:
            print(key)
        assert (match)
        o_value = match.group(2)
        d_value = match.group(3)
        p_value = match.group(4)
        n_value = match.group(5)
        obj_value = match.group(6)
        g_value = match.group(9)  # guide_step
        if g_value is None:
            g_value = 0
        c_value = match.group(11)  # apply_constraint()?
        if c_value is None:
            c_value = 0

        key = key.rsplit('-v2')[0].rsplit('-S64')[0].split(F'{run}-')[1]
        with open(p, 'rb') as fp:
            metric = pickle.load(fp)

        metrics = metricss[F'{run}-G{g_value}-C{c_value}-P{p_value}']
        for k, v in metric.items():
            metrics[key][k].append(dcn(v))

        # key.split('run-270-')[
        metrics[key]['opt'] = int(o_value)
        metrics[key]['diff'] = int(d_value)
        metrics[key]['opt_iter'] = int(n_value)
        metrics[key]['domain'] = int(obj_value)
        metrics[key]['guide_step'] = int(g_value)
        metrics[key]['apply_cons'] = int(c_value)
        if not metrics[key]['opt']:
            metrics[key]['opt_dt'].append(0.0)

    ensure_directory(cfg.out_path)
    for run, metrics in metricss.items():
        filename = F'{cfg.out_path}/metrics-{run}.pkl'
        with open(filename, 'wb') as fp:
            print(F'save {run} to {filename}')
            pickle.dump(dict(metrics), fp)


if __name__ == '__main__':
    main()
