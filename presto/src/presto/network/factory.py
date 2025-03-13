#!/usr/bin/env python3

from typing import Optional, Union, Tuple
from dataclasses import dataclass, replace
from functools import partial
from icecream import ic
import math

from diffusers import (
    DDPMScheduler,
    DDIMScheduler as _DDIMScheduler,
)
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

import torch as th
import torch.nn as nn

from presto.util.torch_util import dcn
from presto.network.dit import DiT
from presto.network.dit_cloud import DiTCloud


def _betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    elif alpha_transform_type == "sqrt_cos":
        def alpha_bar_fn(t):
            return math.cos((t + 0.002) / 1.002 * math.pi / 2) ** 0.5
    else:
        raise ValueError(
            f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return th.tensor(betas, dtype=th.float32)


class LerpFunc:
    def __init__(self, x):
        self.x = x

    def __call__(self, i):
        x = self.x
        i = th.as_tensor(i)
        i0 = th.floor(i).clamp(0, x.shape[-1] - 1).long()
        i1 = (i0 + 1).clamp(0, x.shape[-1] - 1)
        w = (i - i0).float()
        return th.lerp(x[..., i0],
                       x[..., i1],
                       w)


class LerpTensor:
    def __init__(self, x: th.Tensor):
        self.x = x
        self.__lerp_func = LerpFunc(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i: th.Tensor):
        if isinstance(i, int):
            return self.x[i]
        if isinstance(i, th.Tensor) and (i.dtype == th.long):
            return self.x[i]
        return self.__lerp_func(i)


class DDIMScheduler2(_DDIMScheduler):
    def set_timesteps(self, num_inference_steps: int,
                      device: Union[str, th.device] = None):
        out = super().set_timesteps(num_inference_steps, device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(device)
        return out

    def _get_variance(self, timestep, prev_timestep):
        # print('timestep', timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t = alpha_prod_t[..., None, None]

        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep % len(self.alphas_cumprod)
        ].detach().clone()
        alpha_prod_t_prev[prev_timestep < 0] = self.final_alpha_cumprod
        alpha_prod_t_prev = alpha_prod_t_prev[..., None, None]

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (
            beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def _step(
        self,
        model_output: th.FloatTensor,
        timestep: int,
        sample: th.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[th.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`th.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`th.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`th.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep, ..., None, None]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep % len(
            self.alphas_cumprod), ..., None, None].detach().clone()
        alpha_prod_t_prev[prev_timestep < 0] = self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) *
                            pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (
                alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`")

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in
            # Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) *
                            pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (
            1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (
            0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`.")

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype)
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def step(self,
             *args,
             num_train_timesteps: th.Tensor = None,
             num_inference_steps: th.Tensor = None,
             **kwds):
        # cache old results
        # alphas_cumprod_old = self.alphas_cumprod
        if num_inference_steps is not None:
            num_inference_steps_old = self.num_inference_steps
        if num_train_timesteps is not None:
            num_train_timesteps_old = self.config.num_train_timesteps

        # override with args/new
        # self.alphas_cumprod = LerpTensor(alphas_cumprod_old)
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
        if num_train_timesteps is not None:
            self.config.num_train_timesteps = num_train_timesteps
        # print(self.num_inference_steps,
        #       self.config.num_train_timesteps)

        out = self._step(*args, **kwds)

        # restore
        # self.alphas_cumprod = alphas_cumprod_old
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps_old
        if num_train_timesteps is not None:
            self.config.num_train_timesteps = num_train_timesteps_old
        return out


# DDIMScheduler = _DDIMScheduler
DDIMScheduler = DDIMScheduler2


@dataclass
class ModelConfig:
    model_type: str = 'dp'

    # Shared configs that
    # all models have to follow:
    seq_len: int = 1000  # max length of input trajectory
    obs_dim: int = 7  # input/output size
    cond_dim: int = 104  # size of conditioning vector
    embed_dim: int = 256  # size of embedding output

    dit: DiT.Config = DiT.Config()
    dit_cloud: DiTCloud.Config = DiTCloud.Config()

    subnet_type: str = 'dit'

    def __post_init__(self):
        """
        Make sub-configurations consistent
        with the global specification.
        """
        self.dit.input_size = self.seq_len
        self.dit.in_channels = self.obs_dim
        self.dit.hidden_size = self.embed_dim
        self.dit.cond_dim = self.cond_dim

        self.dit_cloud.input_size = self.seq_len
        self.dit_cloud.in_channels = self.obs_dim
        self.dit_cloud.hidden_size = self.embed_dim
        self.dit_cloud.cond_dim = self.cond_dim


@dataclass
class DiffusionConfig:
    num_train_diffusion_iter: int = 1024
    pred_type: str = 'v_prediction'
    sched_type: str = 'ddpm'
    rescale_betas_zero_snr: bool = False
    timestep_spacing: str = 'leading'
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = 'linear'
    clip_sample: bool = False
    shift: float = 3.0


def get_scheduler(cfg):
    if cfg.sched_type == 'ddpm':
        sched = DDPMScheduler(
            num_train_timesteps=cfg.num_train_diffusion_iter,
            clip_sample=cfg.clip_sample,
            prediction_type=cfg.pred_type
        )
    elif cfg.sched_type == 'ddim':
        betas = None
        if cfg.beta_schedule in ['sqrt_cos']:
            betas = dcn(_betas_for_alpha_bar(
                cfg.num_train_diffusion_iter,
                alpha_transform_type=cfg.beta_schedule))
        sched = DDIMScheduler(
            num_train_timesteps=cfg.num_train_diffusion_iter,
            clip_sample=cfg.clip_sample,
            prediction_type=cfg.pred_type,
            rescale_betas_zero_snr=cfg.rescale_betas_zero_snr,
            timestep_spacing=cfg.timestep_spacing,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            trained_betas=betas
        )
    else:
        raise ValueError(F'Unknown sched_type={cfg.sched_type}')
    return sched


def get_model(cfg: ModelConfig, device: Optional[str] = None):
    ic(cfg.model_type)
    if cfg.model_type == 'dit':
        model = DiT(cfg.dit)
    elif cfg.model_type == 'dit_cloud':
        model = DiTCloud(cfg.dit_cloud)
    model = model.to(device)
    # model = th.compile(model)
    return model


def test_lerptensor():
    lt = LerpTensor(th.randn(size=(8, 3)))
    print(lt[2.5])
