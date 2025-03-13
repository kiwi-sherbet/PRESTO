#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from pathlib import Path

from presto.util.config import with_oc_cli_v2


@dataclass
class Config:
    metric: str = 'suc_rate'  # or `avg_pen_cost`
    include_trajopt_results: bool = True
    filter_opt_iter: bool = True
    in_path: str = '/tmp/metrics'
    out_path: str = './save/eval/proc_{:04d}.pkl'
    plot:bool=False


@with_oc_cli_v2
def main(cfg: Config):
    print(cfg.metric)
    proc = {}
    data = {}

    # NOTE(ycho): you may replace `metrics` dict with
    # your preferred alias here; e.g.,
    # metrics = {'run-XXX' : 'alias'}
    metrics = {}
    for f in Path(cfg.in_path).glob('metrics-*.pkl'):
        key = str(f).split('metrics-')[-1].split('.pkl')[0]
        metrics[key] = key

    ks = sorted(metrics.keys())

    def find_key(k: int):
        if k == 0:
            # NOTE(ycho): decide whether to include
            # `trajopt`
            if cfg.include_trajopt_results:
                kk = 'trajopt'
            else:
                return None
        else:
            kk = 'unknown'
            for ii, kk in enumerate(ks):
                if k == (ii + 1):
                    kk = metrics[kk]
                    break
        return kk

    for i, k in enumerate(ks):
        tag = metrics[k]
        with open(F'{cfg.in_path}/metrics-{k}.pkl', 'rb') as fp:
            data_k = pickle.load(fp)
        dd = {F'{tag}_{k}': v for (k, v) in data_k.items()}
        for v in dd.values():
            if v['diff'] == 1:
                v['diff'] = (1 + i)
        data.update(dd)

    def op_if(v, vv, kk, op=np.mean, k=None):
        vv = np.asarray(vv)
        if len(vv.shape) == 1:
            # NOTE(ycho): Two things are happening here:
            # (1) Skip index=0, since iter=0 includes
            # startup overheads (GPU device init/cache alloc)
            # (2) Take the first 180 elements from the metrics data
            # to equalize the number of data-points per domain.
            vv = vv[..., 1:181]
        return op(vv)

    data = {k: {kk: op_if(v, vv, kk, np.nanmean, k=k)
                for (kk, vv) in v.items()} for (k, v) in data.items()}
    df = pd.DataFrame(data.values())

    # Sanitize opt_iter/dt parameters.
    # - opt_iter is zero if optimization is disabled
    # - dt is a sum of:
    # +opt_dt(optimization) +
    # +model_dt(diffusion inference)
    # +pcd_dt(point-cloud encoder)
    df['opt_iter'] = df['opt_iter'] * df['opt']
    df['dt'] = df['opt_dt'] + df['model_dt'] + df['pcd_dt']

    for d, sdf in df.groupby('domain'):
        fig, ax = plt.subplots(1, 3)

        # == populate `proc` dict to export data ==
        for k, ssdf in sdf.groupby('diff'):
            kk = find_key(k)
            if kk is None:
                continue
            ssdf = ssdf.sort_values('dt')
            # NOTE(ycho): we also populate `proc` dict here
            # for explorting.
            proc[F'domain={d},diff={k}'] = dict(
                dt=ssdf['dt'].to_numpy(),
                suc_rate=ssdf['suc_rate'].to_numpy(),
                avg_pen_cost=ssdf['avg_pen_cost'].to_numpy(),
                col_rate=ssdf['col_rate'].to_numpy(),
                # NOTE(ycho): optionally log standard deviations.
                # suc_rate_std=ssdf['suc_rate_std'].to_numpy(),
                # avg_pen_cost_std=ssdf['avg_pen_cost_std'].to_numpy(),
                # col_rate_std=ssdf['col_rate_std'].to_numpy(),
                opt_iter=ssdf['opt_iter'],
                domain=d,
                method=kk
            )

        # == plot dt-metric ==
        if cfg.plot:
            for k, ssdf in sdf.groupby('diff'):
                kk = find_key(k)
                if kk is None:
                    continue

                ssdf = ssdf.sort_values('dt')
                if cfg.filter_opt_iter:
                    src = ssdf[np.in1d(ssdf['opt_iter'], [0, 1, 2, 4, 8])]
                else:
                    src = ssdf
                src.plot(
                    'dt', cfg.metric,
                    legend=True,
                    marker='x',
                    ax=ax[0],
                    label=F'{kk}')
                ax[0].grid()
            ax[0].set_title(F'Domain={int(d)} (dt)')

            # == plot opt_iter-metric ==
            for k, ssdf in sdf.groupby('diff'):
                kk = find_key(k)
                if kk is None:
                    continue

                ssdf = ssdf.sort_values('opt_iter')
                if cfg.filter_opt_iter:
                    src = ssdf[np.in1d(ssdf['opt_iter'], [0, 1, 2, 4, 8])]
                else:
                    src = ssdf
                src.plot(
                    'opt_iter', cfg.metric,
                    legend=True,
                    marker='x',
                    ax=ax[1],
                    label=F'{kk}')
                ax[1].grid()

            # == plot opt_iter-avg_pen_cost ==
            for k, ssdf in sdf.groupby('diff'):
                kk = find_key(k)
                if kk is None:
                    continue

                ssdf = ssdf.sort_values('opt_iter')

                if cfg.filter_opt_iter:
                    src = ssdf[np.in1d(ssdf['opt_iter'], [0, 1, 2, 4, 8])]
                else:
                    src = ssdf
                src.plot(
                    'opt_iter',
                    'avg_pen_cost',
                    legend=True,
                    marker='x',
                    ax=ax[2],
                    label=F'{kk}')
                ax[2].grid()
                ax[2].set_title('avg. pen cost')

            ax[1].set_title(F'Domain={int(d)} (opt. iter)')
            plt.show()

    out_path = Path(cfg.out_path).parent
    out_path.mkdir(parents=True, exist_ok=True)
    proc_index: int = len(list(out_path.glob('proc*.pkl')))
    out_path = cfg.out_path.format(proc_index)

    print(F' == Export `proc` to {out_path} == ')
    with open(out_path, 'wb') as fp:
        pickle.dump(proc, fp)


if __name__ == '__main__':
    main()
