#!/usr/bin/env python3

from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, replace
from tqdm.auto import tqdm

import pickle
import numpy as np
from omegaconf import OmegaConf
from icecream import ic
from contextlib import nullcontext
import time
import math

import warp as wp
import einops
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from presto.data.factory import (DataConfig, get_dataset)
from presto.data.franka_util import franka_fk
from presto.network.common import grad_step, MLP
from presto.network.factory import (ModelConfig, get_model,
                                    DiffusionConfig, get_scheduler)
from presto.network.dit_cloud import DiTCloud
from presto.diffusion.util import pred_x0, diffusion_loss
from presto.cost.curobo_cost import CuroboCost
from presto.cost.cached_curobo_cost import CachedCuroboCost, cached_curobo_cost_with_ng

from presto.util.ckpt import (load_ckpt,
                              save_ckpt,
                              last_ckpt)
from presto.util.path import RunPath, get_path
from presto.util.hydra_cli import hydra_cli
from presto.util.wandb_logger import WandbLogger


# NOTE(ycho): diffusers version compatibility
try:
    from diffusers.training_utils import compute_density_for_timestep_sampling
except ImportError:
    def compute_density_for_timestep_sampling(
        weighting_scheme: str, batch_size: int,
        logit_mean: float = None, logit_std: float = None,
        mode_scale: float = None
    ):
        """Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = th.normal(
                mean=logit_mean, std=logit_std, size=(
                    batch_size,), device="cpu")
            u = th.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = th.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (th.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = th.rand(size=(batch_size,), device="cpu")
        return u


def _map_device(x, device):
    """ Recursively map tensors in a dict to a device.  """
    if isinstance(x, dict):
        return {k: _map_device(v, device) for (k, v) in x.items()}
    if isinstance(x, th.Tensor):
        return x.to(device=device)
    return x


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

    lr_warmup_steps: int = 500
    lr_schedule: str = 'cos'

    num_epochs: int = 512
    batch_size: int = 256
    num_configs: int = 1000
    save_epoch: int = 1
    use_amp: Optional[bool] = None

    collision_coef: float = 0.0
    distance_coef: float = 0.0
    diffusion_coef: float = 1.0
    euclidean_coef: float = 0.0

    mp_cost_type: str = 'ddpm'

    cost: CuroboCost.Config = CuroboCost.Config()
    cached_cost: CachedCuroboCost.Config = CachedCuroboCost.Config()
    use_cached: bool = True
    use_ng: bool = False

    log_by: str = 'epoch'
    step_max: Optional[int] = None
    check_cost: bool = False

    # FOR flow-matching scheduling
    weighting_scheme: str = 'logit_normal'
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29


@dataclass
class MetaConfig:
    project: str = 'trajopt'
    task: str = 'test'
    group: Optional[str] = None
    # NOTE(ycho): if `run_id` is None,
    # then we'll ask for the run_id via `input()` during runtime
    # if `wandb` is enabled.
    run_id: Optional[str] = None
    resume: bool = False


@dataclass
class Config:
    # Global config
    meta: MetaConfig = MetaConfig()
    path: RunPath.Config = RunPath.Config(root='/tmp/presto')
    train: TrainConfig = TrainConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    device: str = 'cuda:0'
    cfg_file: Optional[str] = None
    x0_type: str = 'step'
    x0_iter: int = 4  # iterate 4-steps
    load_ckpt: Optional[str] = None
    reweight_loss_by_coll: bool = False
    use_wandb: bool = True

    @property
    def need_traj(self):
        # We need (unnormalized) joint trajectories only
        # when evaluating auxiliary collision/distance costs.
        return (
            self.train.collision_coef > 0
            or self.train.distance_coef > 0
            or self.reweight_loss_by_coll
            or self.train.euclidean_coef > 0
        )


def collate_fn(xlist):
    cols = [x.pop('col-label') for x in xlist]
    out = th.utils.data.default_collate(xlist)
    out['col-label'] = np.stack(cols, axis=0)
    return out


def train_loop(cfg: Config,
               path: RunPath,
               dataset,
               model: nn.Module,
               sched: SchedulerMixin,
               cost: CuroboCost,
               optimizer: th.optim.Optimizer,
               lr_scheduler: th.optim.lr_scheduler._LRScheduler,
               scaler: th.cuda.amp.GradScaler,
               writer: Any,
               cost_v: Optional[CuroboCost] = None
               ):
    device = cfg.device
    _collate_fn = collate_fn
    loader = th.utils.data.DataLoader(dataset,
                                      batch_size=cfg.train.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=_collate_fn)

    # NOTE(ycho): in case the prediction iterates through
    # multiple steps, we cache the evenly spaced diffusion intervals
    # up to the step corresponding to the noise lvels.
    if cfg.x0_type == 'iter':
        # Cache all possible substeps shorter than `num_train_diffusion_iter`.
        substepss = [
            th.round(
                th.arange(i, 0, -i / cfg.x0_iter, device=cfg.device)).long()
            for i in range(1, cfg.diffusion.num_train_diffusion_iter + 1)]
        substepss = th.stack(substepss, dim=0).to(
            device=cfg.device,
            dtype=th.long).sub_(1)

    try:
        diff_step_max: int = sched.config.num_train_timesteps
        if cfg.train.step_max is not None:
            diff_step_max = cfg.train.step_max

        global_step = 0
        for epoch in tqdm(range(cfg.train.num_epochs),
                          desc=path.dir.name):
            loss_ddpm_ra = []
            loss_coll_ra = []
            loss_dist_ra = []
            loss_eucd_ra = []

            for batch in tqdm(loader, leave=False):
                # Transfer _batch_ to target device
                batch = _map_device(batch, device=cfg.device)
                # Batch content:
                # trajectory: ground truth joint trajectory
                # label: env. representation for model conditioning
                # col: explicit env. representation for cost evaluation
                # now looks like (..., C, S)
                true_traj_ds = batch['trajectory'].swapaxes(-1, -2)

                label = batch['env-label']
                if cfg.train.use_cached:
                    col = batch['prim-label']
                else:
                    col = batch['col-label']

                # Add noise and sample timesteps.
                steps = th.randint(
                    0, diff_step_max,
                    (true_traj_ds.shape[0],),
                    device=device
                ).long()
                noise = th.randn(true_traj_ds.shape, device=device)
                noisy_trajs = sched.add_noise(true_traj_ds, noise, steps)

                # Compute model predictions.
                preds = model(noisy_trajs,
                              steps,
                              return_dict=False,
                              class_labels=label)[0]

                # Evaluate Diffusion loss and log.
                loss_ddpm = diffusion_loss(
                    preds, true_traj_ds,
                    noise, noisy_trajs, steps,
                    sched=sched,
                    reduction=('none' if cfg.reweight_loss_by_coll
                               else 'mean')
                )

                # Add extra collision/distance costs.
                loss_coll = th.zeros_like(loss_ddpm.mean())
                loss_dist = th.zeros_like(loss_ddpm.mean())

                # Compute the trajectory at t=0 based on x0_type.
                if cfg.need_traj:
                    if cfg.x0_type == 'legacy':
                        # Technically problematic?
                        # previous combination:
                        # `epsilon` + traj=preds(=epsilon)
                        pred_traj_ds = preds
                        raise ValueError(
                            'This option is broken and no longer supported.'
                        )
                    elif cfg.x0_type == 'step':
                        # This is for DDIM
                        pred_traj_ds = pred_x0(
                            sched, steps, preds, noisy_trajs)
                    elif cfg.x0_type == 'iter':
                        # Arrive at `traj@t=0` through multiple (DDIM)
                        # iterations.
                        substeps = substepss[steps]
                        pred_traj_ds = noisy_trajs
                        if (isinstance(sched, DDIMScheduler)):
                            for i in range(cfg.x0_iter):
                                preds = model(pred_traj_ds,
                                              substeps[..., i],
                                              return_dict=False,
                                              class_labels=label)[0]
                                pred_traj_ds = sched.step(
                                    preds, substeps[..., i], pred_traj_ds,
                                    num_train_timesteps=steps,
                                    num_inference_steps=cfg.x0_iter).prev_sample
                        else:
                            raise ValueError(F'Invalid scheduler: {sched}')

                    # u_traj: Unnormalized trajectory.
                    # needed for collision/distance cost evaluations.
                    # It also swaps the channel dimension and the sequence
                    # dimension (...CS -> ...SC).
                    u_pred_sd = dataset.normalizer.unnormalize(
                        pred_traj_ds.swapaxes(-1, -2))

                # Optionally evaluate collision and distance costs.
                if cfg.train.collision_coef > 0:
                    assert (u_pred_sd.shape[-1] == dataset.obs_dim)
                    assert (u_pred_sd.shape[-2] == dataset.seq_len)
                    if cfg.train.use_ng:
                        # numerical gradient
                        loss_coll = cached_curobo_cost_with_ng(
                            u_pred_sd,
                            lambda q: cost(q, col)).mean()
                    else:
                        # analytical gradient
                        loss_coll = cost(u_pred_sd, col).mean()

                    if cost_v is not None:
                        # evaluate validation cost
                        with th.no_grad():
                            loss_coll_v = []
                            for i in range(u_pred_sd.shape[0]):
                                loss_coll_v_i = cost_v(
                                    u_pred_sd[i: i + 1],
                                    # col[i: i + 1]
                                    {k: v[i:i + 1] for (k, v) in col.items()}
                                )
                                loss_coll_v.append(loss_coll_v_i)
                            loss_coll_v = th.cat(loss_coll_v, dim=0)
                            print((loss_coll - loss_coll_v).mean())

                if cfg.train.distance_coef > 0:
                    # penalize distance between adjacent configurations.
                    assert (u_pred_sd.shape[-2] == dataset.seq_len)
                    loss_dist = F.mse_loss(u_pred_sd[..., 1:, :],
                                           u_pred_sd[..., :-1, :])

                if cfg.train.euclidean_coef > 0:
                    x_pred_sd = franka_fk(u_pred_sd)
                    with th.no_grad():
                        x_true_sd = franka_fk(
                            dataset.normalizer.unnormalize(batch['trajectory'])
                        )
                    loss_eucd = F.mse_loss(x_pred_sd, x_true_sd)

                if cfg.reweight_loss_by_coll:
                    with th.no_grad():
                        u_pred_sd = dataset.normalizer.unnormalize(
                            pred_traj_ds.swapaxes(-1, -2))
                        # in general `weight` will be 1/eta * (d+eta)^2
                        # where eta~0.1, so 10 * (d+0.1)^2
                        # == maxing out at `eta` since we assume
                        # `d<=0` for the ground-truth trajectories.
                        weight = 1.0 + cost(u_pred_sd, col) / cost.cfg.margin
                        weight /= weight.sum(dim=-1, keepdim=True)
                    loss_ddpm = loss_ddpm.mean(
                        dim=-2).mul(weight).sum(dim=-1).mean()
                else:
                    loss_ddpm = loss_ddpm.mean()

                # Aggregate loss terms.
                loss = (cfg.train.diffusion_coef * loss_ddpm
                        + cfg.train.collision_coef * loss_coll
                        + cfg.train.distance_coef * loss_dist
                        + cfg.train.euclidean_coef * loss_eucd
                        )

                # Take gradient step.
                grad_step(loss, optimizer, scaler=scaler)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                global_step += 1

                if cfg.train.log_by == 'step':
                    with th.no_grad():
                        train_step_log = {'step': global_step,
                                          'loss_ddpm': loss_ddpm.item(),
                                          'loss_collision': loss_coll.item(),
                                          'loss_distance': loss_dist.item(),
                                          'loss_euclidean': loss_eucd.item(),
                                          'loss': loss.item()
                                          }
                        if lr_scheduler is not None:
                            train_step_log['lr'] = lr_scheduler.get_last_lr()[
                                0]
                        else:
                            # NOTE(ycho): may not _always_ work.
                            train_step_log['lr'] = optimizer.param_groups[0][
                                'lr']
                        writer.log_train_data(
                            {'Train/{}'.format(key): item for key,
                             item in train_step_log.items()},
                            epoch=global_step)

                with th.no_grad():
                    loss_coll_ra.append(loss_coll.item())
                    loss_dist_ra.append(loss_dist.item())
                    loss_ddpm_ra.append(loss_ddpm.item())
                    loss_eucd_ra.append(loss_eucd.item())

            # Logging & bookkeeping
            if cfg.train.log_by == 'epoch':
                mean_loss_ddpm = sum(loss_ddpm_ra) / len(loss_ddpm_ra)
                mean_loss_collision = sum(loss_coll_ra) / len(loss_coll_ra)
                mean_loss_distance = sum(loss_dist_ra) / len(loss_dist_ra)
                mean_loss_euclidean = sum(loss_eucd_ra) / len(loss_eucd_ra)
                mean_loss_sum = (
                    cfg.train.diffusion_coef * mean_loss_ddpm
                    + cfg.train.collision_coef * mean_loss_collision
                    + cfg.train.distance_coef * mean_loss_distance
                    + cfg.train.euclidean_coef * mean_loss_euclidean
                )
                train_step_log = {'step': global_step,
                                  'loss_ddpm': mean_loss_ddpm,
                                  'loss_collision': mean_loss_collision,
                                  'loss_distance': mean_loss_distance,
                                  'loss_euclidean': mean_loss_euclidean,
                                  'loss': mean_loss_sum}
                if lr_scheduler is not None:
                    train_step_log['lr'] = lr_scheduler.get_last_lr()[0]
                if (writer is not None):
                    writer.log_train_data(
                        {'Train/{}'.format(key): item for key,
                         item in train_step_log.items()},
                        epoch=epoch)

            # Periodically save checkpoints.
            if (epoch % cfg.train.save_epoch) == 0:
                save_dict = {'model': model,
                             'optimizer': optimizer,
                             'normalizer': dataset.normalizer}
                if lr_scheduler is not None:
                    save_dict['lr_scheduler'] = lr_scheduler
                save_ckpt(save_dict,
                          ckpt_file=F'{path.ckpt}/{epoch:03d}.ckpt')
    finally:
        save_dict = {'model': model,
                     'optimizer': optimizer,
                     'normalizer': dataset.normalizer}
        if lr_scheduler is not None:
            save_dict['lr_scheduler'] = lr_scheduler
        save_ckpt(save_dict,
                  ckpt_file=F'{path.ckpt}/last.ckpt')


def train(cfg: Config):
    device: str = cfg.device

    # basic layout = B,T,D
    train_dataset = get_dataset(cfg.data, 'train',
                                # NOTE(ycho): direct GPU-load is
                                # infeasible for larger datasets
                                # device=device
                                device='cpu')

    # Update model config based on dataset specification.
    obs_dim = train_dataset.obs_dim
    seq_len = train_dataset.seq_len

    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            obs_dim=obs_dim,
            seq_len=seq_len,
            cond_dim=train_dataset.cond_dim)
    )

    # Initialize the experiment, and save the configuration.
    path = RunPath(cfg.path)
    ic(cfg)

    # Create the model and scheduler.
    model = get_model(cfg.model, device=cfg.device)
    sched = get_scheduler(cfg.diffusion)
    ic(model)
    ic(sched)

    # Create (wandb) logger.
    if cfg.use_wandb:
        if cfg.meta.run_id is None:
            run_id = input('Name this run:')
            if not cfg.meta.resume:
                # _only_ in case of a new run, we
                # automatically append the runpath to
                # the run name.
                run_id = F'{run_id}-{path.dir.name}'
            cfg = replace(cfg,
                          meta=replace(cfg.meta,
                                       run_id=run_id))
        wandb_logger = WandbLogger(
            project=cfg.meta.project,
            group=cfg.meta.group,
            id=cfg.meta.run_id,
            task=cfg.meta.task,
            path=path.tb_log,
            config=OmegaConf.to_container(
                OmegaConf.structured(cfg)),
            model=model
        )
    else:
        wandb_logger = None
    OmegaConf.save(cfg, F'{path.dir}/cfg.yaml')

    sched.set_timesteps(
        cfg.diffusion.num_train_diffusion_iter,
        device=cfg.device)

    # In case of DDIM, configure devices for extra cached tensors.
    # NOTE(ycho): not sure if necessary
    if isinstance(sched, DDIMScheduler):
        sched.alphas_cumprod = sched.alphas_cumprod.to(cfg.device)
        sched.final_alpha_cumprod = sched.final_alpha_cumprod.to(cfg.device)

    # In case trajopt cost is enabled, we'll add
    # auxilirary loss terms from Curobo.
    # cost - cost for training
    # cost_v - cost for validation
    cost_v = None
    if cfg.train.use_cached:
        cost = CachedCuroboCost(cfg.train.cached_cost,
                                batch_size=cfg.train.batch_size,
                                device=cfg.device,
                                n_prim=train_dataset._num_prims,
                                )
        # NOTE(ycho): _only_ for validation
        if cfg.train.check_cost:
            cost_v = CachedCuroboCost(cfg.train.cached_cost,
                                      batch_size=cfg.train.batch_size,
                                      device=cfg.device,
                                      n_prim=train_dataset._num_prims,
                                      )
    else:
        cost = CuroboCost(cfg.train.cost,
                          batch_size=cfg.train.batch_size,
                          device=cfg.device)
        if cfg.train.check_cost:
            cost_v = CuroboCost(cfg.train.cost,
                                batch_size=cfg.train.batch_size,
                                device=cfg.device)
    ic(model)
    ic(sched)

    # Create the optimizer and learning rate scheduler.
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=cfg.train.learning_rate,
                               weight_decay=cfg.train.weight_decay)

    N = len(train_dataset)
    num_training_steps = int(
        (N / cfg.train.batch_size) *
        cfg.train.num_epochs)

    if cfg.train.lr_schedule == 'cos':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.train.lr_warmup_steps,
            num_training_steps=num_training_steps)
    else:
        lr_scheduler = None

    if cfg.load_ckpt is not None:
        load_dict = {'model': model,
                     'optimizer': optimizer}
        if lr_scheduler is not None:
            load_dict['lr_scheduler'] = lr_scheduler
        load_ckpt(load_dict, last_ckpt(F'{cfg.load_ckpt}'))

    # Create gradient scaling
    # in case of mixed-precision training.
    scaler = th.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
    amp_ctx = (nullcontext() if (cfg.train.use_amp is None)
               else th.cuda.amp.autocast(enabled=cfg.train.use_amp))

    with amp_ctx:
        train_loop(cfg, path, train_dataset, model, sched, cost,
                   optimizer, lr_scheduler, scaler, wandb_logger,
                   cost_v=cost_v)


@hydra_cli(
    # NOTE(ycho): you may need to configure `config_path`
    # depending on where you run this script.
    # config_path='presto/src/presto/cfg/',
    config_name='train_v2')
def main(cfg: Config):
    cfg0 = OmegaConf.structured(Config())
    cfg0.merge_with(cfg)
    cfg = cfg0
    if cfg.cfg_file is not None:
        cfg.merge_with(OmegaConf.load(cfg.cfg_file))
        cfg.merge_with_cli()
    cfg = OmegaConf.to_object(cfg)

    wp.init()
    if hasattr(wp, 'ScopedMempool'):
        with wp.ScopedMempool(cfg.device, True):
            train(cfg)
    else:
        train(cfg)


if __name__ == '__main__':
    main()
