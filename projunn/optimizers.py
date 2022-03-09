'''
File: optimizers.py
Created Date: Wed Mar 09 2022
Author: Randall Balestriero
-----
Last Modified: Wed Mar 09 2022 3:47:45 AM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
'''
from typing import Tuple, List
from numpy import iterable
import torch
from torch.optim import Optimizer
import logging
from . import utils


def split_params(
    parameters: iterable, key: str = "needs_projections"
) -> Tuple[List, List]:
    """Takes a list (or iterable) of parameters, and return two lists, the first
    that are all the parameters that do not have ``key'' as an attribute, the second
    are the parameters with ``key'' as an attribute.

    Args:
        parameters (iterable): list of parameters to split
        key (str, optional): The name of the attribute used for splitting. Defaults to "needs_projections".

    Returns:
        Tuple[List, List]: the split parameters
    """
    polar_parameters = []
    standard_parameters = []
    for param in parameters:
        if hasattr(param, "needs_projection"):
            polar_parameters.append(param)
        else:
            standard_parameters.append(param)
    return standard_parameters, polar_parameters


class RMSprop(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        projector,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
        )
        super(RMSprop, self).__init__(params, defaults)

        self.projector = projector

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                square_avgs.append(state["square_avg"])

                if group["momentum"] > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                if group["centered"]:
                    grad_avgs.append(state["grad_avg"])

                state["step"] += 1

        for i, param in enumerate(params_with_grad):
            grad = grads[i]
            square_avg = square_avgs[i]

            if group["weight_decay"] != 0:
                grad = grad.add(param, alpha=group["weight_decay"])

            square_avg.mul_(group["alpha"]).addcmul_(
                grad, grad, value=1 - group["alpha"]
            )

            if group["centered"]:
                grad_avg = grad_avgs[i]
                grad_avg.mul_(group["alpha"]).add_(grad, alpha=1 - group["alpha"])
                avg = (
                    square_avg.addcmul(grad_avg, grad_avg, value=-1)
                    .sqrt_()
                    .add_(group["eps"])
                )
            else:
                avg = square_avg.sqrt().add_(group["eps"])

            if group["momentum"] > 0:
                buf = momentum_buffer_list[i]
                buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                update = -group["lr"] * buf
            else:
                update = -group["lr"] * grad / avg
            if hasattr(param, "needs_projection"):
                update = self.projector(param, update)
            param.add_(update)


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
    """

    def __init__(
        self,
        params,
        projector,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
    ):
        self.projector = projector
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            for i, param in enumerate(params_with_grad):

                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                alpha = lr if maximize else -lr
                update = alpha * d_p
                if hasattr(param, "needs_projection"):
                    update = self.projector(param, update)
                param.add_(update)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss
