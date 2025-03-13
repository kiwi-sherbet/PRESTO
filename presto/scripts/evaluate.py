#!/usr/bin/env python3

from dataclasses import dataclass, replace
from omegaconf import OmegaConf
from typing import Optional, Union, List
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler
)
from curobo.util.logger import setup_curobo_logger

import pickle
import json
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import time

from presto.util.torch_util import dcn
from presto.network.factory import (get_model, get_scheduler)

from presto.diffusion.presto_pipeline import (
    EtudePipeline,
    EtudeGenerator,
    visualize_robot_motion,
    visualize_diffusion_chain
)
from presto.cost.curobo_cost import CuroboCost
from presto.data.factory import (DataConfig, get_dataset)
from presto.data.normalize import Normalize
from presto.util.path import ensure_directory
from presto.util.ckpt import (load_ckpt, last_ckpt)
from presto.diffusion.util import pred_x0, diffusion_loss

from icecream import ic

from train import Config as TrainConfig


@dataclass
class Config(TrainConfig):
    pipeline: EtudePipeline.Config = EtudePipeline.Config(
        n_guide_step=0,
        n_denoise_step=1,
        guide_start=50,
        # cost=CuroboCost.Config(
        #    margin=0.0,
        # ),
        post_guide=0,
        init_type='random',
        guide_scale=0.00005
    )

    show: str = 'none'
    show_at_iter: int = -1

    export_dir: Optional[str] = None
    load_dir: Optional[str] = None
    load_ckpt: Optional[str] = None
    traj_save_path: Optional[str] = None
    eval_save_path: Optional[str] = None
    num_eval: int = 1
    diff_step: Optional[int] = None
    load_normalizer: str = 'data'
    shuffle: bool = True

    filter_file: str = 'data/etude_cabinet_eval/filtered_idx.json'
    skip_filtered: bool = False
    use_cloud: bool = False


def evaluate(
        cfg: Config,
        eval_step: int,
        dataset,
        pipeline: EtudePipeline,
        batch_size: int = 1,
        shuffle: bool = True,
        offset: int = 0,
        step: Optional[int] = None,
        show: str = 'motion',
        export_dir: Optional[str] = None,
        normalizer=None):
    """
    Args:
        cfg:
            Config instance.
        dataset:
            Dataset instance (e.g. EtudeDataset1Sphere)
        pipeline:
            EtudePipeline instance.
        batch_size:
            Number of different scenes, for which the trajectory will be sampled.
            this is different from `cfg.pipeline.expand` which determines the
            number of diffusion seeds shared for a single scene!
        shuffle:
            Whether to shuffle the dataset.
        offset:
            The offset at which the sampled batches will start.
            only used when shuffle=False.
        step:
            number of diffusion steps; by default,
            set to `cfg.diffusion.num_train_diffusion_iter`.
        show:
            visualization type for the diffused trajectory.
            One of the following options are supported:
            motion:
                Shows batches of robot configurations
                moving across space over time-steps.
                Generally a good idea to use batch_size=1 for this
                to avoid visual confusion.
            chain:
                Shows entire trajectory evolving over diffusion-steps.
            none:
                Do not visualize the motions.
        export_dir:
            The directory at which the diffusion visualization images
            will be saved. Images are not saved if None.
    """

    # == Infer Trajectory ==
    if step is None:
        step = cfg.diffusion.num_train_diffusion_iter

    t0 = time.time()
    output = pipeline(
        data_fn=EtudeGenerator(dataset,
                               batch_size,
                               shuffle=shuffle,
                               offset=offset,
                               init_type=cfg.pipeline.init_type,
                               expand=cfg.pipeline.expand,
                               index=cfg.pipeline.index,
                               cond_type=cfg.pipeline.cond_type),
        num_inference_steps=step
    )
    if cfg.eval_save_path is not None:
        eval_save_path = ensure_directory(cfg.eval_save_path)
        with open(eval_save_path / F'{eval_step:03d}.pkl', 'wb') as fp:
            pickle.dump(output['metric'], fp)
    t1 = time.time()

    if normalizer is None:
        normalizer = dataset.normalizer
    output['trajs'] = normalizer.unnormalize(
        output['trajs'].swapaxes(-1, -2)).swapaxes(-1, -2)

    if 'true_traj' in output:
        output['true_traj'] = normalizer.unnormalize(
            output['true_traj']
        )

    # Optionally also save the entire output of the pipeline.
    # This is primarily for debugging.
    if cfg.traj_save_path is not None:
        with open(cfg.traj_save_path, 'wb') as fp:
            pickle.dump({k: dcn(v) for (k, v)
                        in output.items()}, fp)

    # S B D T -> S B T D
    diffusion_chain = output['trajs'].swapaxes(-1, -2)

    # Optionally visualize the point cloud of the scene.
    # This option is mostly unused for now.
    cloud = None
    if 'cloud' in output:
        cloud = output['cloud']

    if show == 'chain':
        # Visualize endpoint trajectories across
        # diffusion iterations.
        visualize_diffusion_chain(diffusion_chain,
                                  # cond=cond,
                                  # cond=list(output['col-label']),
                                  cond=output['col-label'],
                                  cloud=cloud,
                                  cond_type='curobo'
                                  )
    elif show == 'motion':
        # Visualize robot motion at the end.
        visualize_robot_motion(
            diffusion_chain[cfg.show_at_iter],
            # NOTE(ycho): The only valid `cond` input for now
            # is the primitive-based obstacle encodings (`col-label`)
            cond=output['col-label'],
            cloud=cloud,
            export_dir=export_dir,
            CUTOFF=1000,
            collision_label=(output['cost'] > 0.0),
            timeout=0,
            true_traj=output['true_traj']
        )
    elif show == 'plot-chain':
        # Animate diffusion chain.
        fig, axs = plt.subplots(7, 1)
        axs[6].set_xlabel('index')
        dc = diffusion_chain
        for i_b in range(diffusion_chain.shape[1]):
            for i_s in range(diffusion_chain.shape[0]):
                traj_i = dcn(diffusion_chain[i_s, i_b])
                true_i = dcn(output['true_traj'][i_b])

                for i in range(7):
                    axs[i].cla()
                    axs[i].plot(traj_i[..., i])
                    axs[i].plot(true_i[..., i], 'k')
                    axs[i].grid()
                    axs[i].set_ylabel(F'q[{i}]')
                axs[0].set_title(F'{i_s}/{diffusion_chain.shape[0]}')
                if i_b == 0:
                    plt.savefig(
                        F'/tmp/docker/presto-diffusion/presto-{i_s:03d}.png')
                plt.pause(0.001)

    elif show == 'plot_2d':
        # Show trajectory as 2D "image"
        # where columns = timesteps,
        # rows = joint indices (0 ~ 7)
        traj = diffusion_chain[cfg.show_at_iter]
        for i_b in range(traj.shape[0]):
            plt.pcolor(dcn(traj[i_b].swapaxes(-1, -2)))
            plt.colorbar()
            plt.show()

    elif show == 'plot':
        # discretization resolution...
        REP: int = 31
        # 0.5 rad, around 30deg
        BOUND: float = 0.5

        # .detach().clone().requires_grad_(True)
        traj = diffusion_chain[cfg.show_at_iter]

        if True:
            traj_ = traj.detach().clone().requires_grad_(True)
            cost_ = pipeline.cost_e(traj_,
                                    output['col-label'],
                                    reduce=False)
            grad_, = th.autograd.grad(cost_,
                                      traj_,
                                      th.ones_like(cost_))
            margin_ = th.where(cost_[..., None] < 0,
                               cost_[..., None] / grad_,
                               th.zeros_like(grad_))

        delta = th.linspace(-BOUND, +BOUND, REP, dtype=traj.dtype,
                            device=traj.device)

        trajs = []
        for i in range(7):
            # tt = einops.repeat(traj, '... s d -> ... r s d', r=REP).clone()
            tt = einops.repeat(
                output['true_traj'],
                '... s d -> ... r s d',
                r=REP).clone()
            tt[..., i] += delta[:, None]
            trajs.append(tt)
        trajs = th.stack(trajs, dim=-4)  # ... d s r d
        trajs = einops.rearrange(trajs, '... d1 r s d2 -> ... (d1 r s) d2')
        cost = pipeline.cost_e(trajs,
                               output['col-label'],
                               reduce=False)
        cost = cost.reshape(*cost.shape[:-1],
                            7,  # d
                            REP,  # r
                            -1  # s
                            )

        # upper/lower bounds
        lo = th.zeros_like(traj)
        hi = th.zeros_like(traj)
        for i in dcn(th.argsort(th.abs(delta))):
            msk = (cost[..., i, :] <= 0).swapaxes(-1, -2)  # 2,7,256
            lo[msk] = th.minimum(lo[msk], delta[i])
            hi[msk] = th.maximum(hi[msk], delta[i])

        fig, axs = plt.subplots(7, 1)
        axs[6].set_xlabel('index')
        for i_b in range(cost.shape[0]):
            cost_b = dcn(cost[i_b])
            traj_b = dcn(traj[i_b])
            true_b = dcn(output['true_traj'][i_b])
            # lo_b = dcn(traj[i_b] + lo[i_b])
            # hi_b = dcn(traj[i_b] + hi[i_b])
            lo_b = dcn(output['true_traj'][i_b] + lo[i_b])
            hi_b = dcn(output['true_traj'][i_b] + hi[i_b])
            for i in range(7):
                # collision-free bounds
                axs[i].fill_between(th.arange(traj_b.shape[0]),
                                    lo_b[..., i],
                                    hi_b[..., i],
                                    alpha=0.5)
                axs[i].plot(traj_b[..., i])
                axs[i].plot(true_b[..., i], 'k')
                axs[i].grid()
                axs[i].set_ylabel(F'q[{i}]')
            plt.show()
    elif show == 'none':
        pass
    else:
        raise ValueError(F'Unknown visualization option: {show}')


def sample(cfg: Config):
    device = cfg.device

    # NOTE(ycho): we currently do not use validation datasets.
    # The evaluation dataset has to be specified explicitly by
    # changing the data-path (from cfg.data)
    test_dataset = get_dataset(cfg.data, 'train', device=device)

    normalizer = None
    if cfg.load_normalizer == 'data':
        normalizer = test_dataset.normalizer
        print(normalizer.center,
              normalizer.radius)
    elif cfg.load_normalizer == 'ckpt':
        # FIXME(ycho): hardcoded DoF
        normalizer = Normalize(th.zeros(7, device=device),
                               th.zeros(7, device=device))
    elif cfg.load_normalizer == 'hack':
        # FIXME(ycho): hardcoded normalization constants
        normalizer = Normalize.from_avgstd(
            th.as_tensor(
                [0.0000, 0.0000, 0.0000, -1.5708, 0.0000, 2.0069, 0.0000],
                dtype=th.float32, device=device),
            th.as_tensor(
                [2.8873, 1.7528, 2.8873, 1.4910, 2.8873, 1.7356, 2.8873],
                dtype=th.float32, device=device))
    else:
        raise ValueError(F'Unknown normalizer opt: {cfg.load_normalizer}')

    # Load model and scheduler.
    # Print the model for validation.
    model = get_model(cfg.model, device=device)
    sched = get_scheduler(cfg.diffusion)
    model.eval()
    ic(model)
    ic(sched)

    ckpt_path: str = None
    if cfg.load_ckpt is not None:
        ckpt_path = last_ckpt(cfg.load_ckpt)
    elif cfg.load_dir is not None:
        ckpt_path = last_ckpt(F'{cfg.load_dir}/ckpt')

    if cfg.load_normalizer == 'ckpt':
        load_ckpt({'model': model,
                   'normalizer': normalizer},
                  ckpt_path)
        print('load-normalizer', normalizer.center, normalizer.radius)
    else:
        load_ckpt({'model': model},
                  ckpt_path)

    if cfg.load_normalizer in ['ckpt', 'hack']:
        # Ensure normalizer statistics are consistent
        # with the model checkpoint (or "hack").
        test_dataset.renormalize_(normalizer)
        print(test_dataset.normalizer.center,
              test_dataset.normalizer.radius)

    pipeline = EtudePipeline(cfg.pipeline,
                             unet=model,
                             scheduler=sched,
                             batch_size=cfg.train.batch_size,
                             n_prim=(
                                 test_dataset._num_prims
                                 if test_dataset.cfg.prim
                                 else None
                             )
                             )
    # Skip "rejected" tasks
    skip = []
    if cfg.skip_filtered:
        with open(cfg.filter_file, 'r') as fp:
            data = json.load(fp)
        for key, task_files in data.items():
            if key not in cfg.data.shelf.dataset_dir:
                continue

            for task_file in task_files:
                # NOTE(ycho):
                # while this index mapping wouldn't be _generally_ correct,
                # this is at least correct based on our
                # current generation scheme.
                task_index = int(Path(task_file).stem) - 1
                skip.append(task_index)
        print('SKIP', skip)

    for i in range(cfg.num_eval):
        if (cfg.skip_filtered) and (i in skip):
            continue
        if cfg.export_dir is not None:
            export_dir = F'{cfg.export_dir}/{i:03d}'
        else:
            export_dir = None
        evaluate(cfg,
                 i,
                 test_dataset,
                 pipeline,
                 batch_size=cfg.train.batch_size,
                 shuffle=cfg.shuffle,
                 offset=(i * cfg.train.batch_size),
                 show=cfg.show,
                 export_dir=export_dir,
                 step=cfg.diff_step,
                 normalizer=normalizer,
                 )


def main():
    # th.autograd.set_detect_anomaly(True)
    # setup_curobo_logger(level='debug')
    cfg = OmegaConf.structured(Config())

    # NOTE(ycho): temporary hack for now to
    # load configs from both CLI-derived files and CLI arguments
    cfg.merge_with_cli()
    cfg.merge_with(OmegaConf.load(F'{cfg.load_dir}/cfg.yaml'))
    cfg.merge_with_cli()

    # FIXME(ycho): some kcfg runs that were
    # supposed to use key configs were actually using positions.
    # We temporarily override those configurations by
    # directly inspecting the `cond_dim` of the model.
    cfg.data.load_cloud = False

    cfg = OmegaConf.to_object(cfg)
    sample(cfg)


if __name__ == '__main__':
    main()
