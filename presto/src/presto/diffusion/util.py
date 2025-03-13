#!/usr/bin/env python3

from typing import Optional
import torch as th
import torch.nn.functional as F
from diffusers import DDIMScheduler


def diffusion_loss(preds: th.Tensor,
                   trajs: th.Tensor,
                   noise: th.Tensor,
                   noisy_trajs: th.Tensor,
                   steps: th.Tensor,
                   # pred_type: str
                   sched: DDIMScheduler,
                   *args, **kwds
                   ):
    """
    Evaluate diffusion loss based on `pred_type`.
    """
    pred_type = sched.config.prediction_type
    if pred_type == 'epsilon':
        loss_ddpm = F.mse_loss(preds, noise, *args, **kwds)
    elif pred_type == 'sample':
        # loss_ddpm = F.mse_loss(preds, noisy_trajs - trajs)
        loss_ddpm = F.mse_loss(preds, trajs, *args, **kwds)
    elif pred_type == 'v_prediction':
        v = sched.get_velocity(trajs, noise, steps)
        loss_ddpm = F.mse_loss(preds, v, *args, **kwds)
    else:
        raise ValueError(F'Unknown pred_type = {pred_type}')
    return loss_ddpm


def pred_x0(sched: DDIMScheduler,
            t: th.Tensor,
            v: th.Tensor,
            x: th.Tensor):
    """
    (Adapted from diffusers.DDIMScheduler.)
    Predict x_0 from (t, v_t, x_t), where:
        * t is the timestep of diffusion,
        * v_t is the model output,
        * x_t is the current (noisy) sample.
    """
    t_prev = t - sched.config.num_train_timesteps // sched.num_inference_steps

    if sched.config.prediction_type in ['epsilon', 'v_prediction']:
        ac = sched.alphas_cumprod.to(t.device)
        alpha_prod_t = ac[t]
        alpha_prod_t_prev = ac[t_prev % len(ac)]
        if hasattr(sched, 'final_alpha_cumprod'):
            # DDIM
            alpha_prod_t_prev[t_prev < 0] = sched.final_alpha_cumprod
        else:
            # DDPM
            alpha_prod_t_prev[t_prev < 0] = sched.one
        alpha_prod_t = alpha_prod_t[..., None, None]
        beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if sched.config.prediction_type == "epsilon":
        pred_original_sample = (
            x - beta_prod_t ** (0.5) * v) / alpha_prod_t ** (0.5)
    elif sched.config.prediction_type == "sample":
        pred_original_sample = v
    elif sched.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * x - (beta_prod_t**0.5) * v
    else:
        raise ValueError(
            f"prediction_type given as {sched.config.prediction_type}" +
            " must be one of `epsilon`, `sample`, or" +
            " `v_prediction`")
    return pred_original_sample
