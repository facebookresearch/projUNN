"""
File: utils.py
Created Date: Tue Mar 15 2022
Author: Randall Balestriero and Bobak Kiani
-----
Last Modified: Tue Mar 15 2022 7:30:04 PM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import torch
import numpy as np


def orthogonal_(tensor, gain=1, real_only=False):
    r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
        real_only: bool

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)
    if real_only and torch.is_complex(flattened):
        flattened = flattened.real

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sgn()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def conjugate_transpose(v):
    return torch.conj(torch.transpose(v, -2, -1))


def LSI_approximation(A, k, eps=1e-2):
    n = A.shape[-1]
    R = torch.empty(n, k, dtype=A.dtype, device=A.device)
    R = orthogonal_(R).type(A.dtype)
    B = np.sqrt(n / k) * conjugate_transpose(R) @ A
    v, e, _ = torch.linalg.svd(B @ conjugate_transpose(B))
    v = conjugate_transpose(B) @ (v / (e.unsqueeze(-2).sqrt() + eps))
    return A @ v, v


def add_outer_products(a, b):
    return a @ conjugate_transpose(b)


def norm_squared(M):
    return torch.sum(M * torch.conj(M), dim=-2, keepdims=True)


def replace_nans_with_Id(A):
    check = torch.isnan(A)
    if torch.any(check):
        check = torch.any(torch.any(check, -1), -1)
        A[check][0, 0] = 1.0
        A[check][1, 1] = 1.0
        A[check][1, 0] = 0.0
        A[check][0, 1] = 0.0
    return A


def normalize_properly(A):
    id = torch.eye(2, dtype=A.dtype, device=A.device)
    check = torch.isclose(A @ conjugate_transpose(A), id, 1e-3, 1e-5)
    if len(A.shape) == 3:
        check = torch.any(torch.any(torch.logical_not(check), -1), -1)
        A[check] = id
    else:
        if torch.any(torch.logical_not(check)):
            A = id
    return A


default_complex_dtype = torch.complex64


def dim2_eig(A):
    # need to check to make sure this works correctly (currently fails for zero input)
    if len(A.shape) == 3:
        batched = True
    else:
        batched = False
    if batched:
        A = A.permute(1, 2, 0)
    det = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]).type(default_complex_dtype)
    trace = (A[0, 0] + A[1, 1]).type(default_complex_dtype)
    eig = torch.stack(
        [
            trace / 2 + torch.sqrt(trace * trace / 4 - det),
            trace / 2 - torch.sqrt(trace * trace / 4 - det),
        ],
        dim=batched * 1,
    )
    vec = torch.stack(
        (
            eig - A[1, 1].unsqueeze(-1).expand(*[-1, 2] if batched else [2]),
            A[1, 0].unsqueeze(-1).expand(*[-1, 2] if batched else [2]),
        ),
        dim=batched * 1 + 1,
    )
    vec = torch.transpose(vec, -2, -1)
    # print(1/torch.sqrt(norm_squared(vec)))
    # check divide by zero
    return eig, normalize_properly(
        vec / torch.sqrt(norm_squared(vec))
    )  # may want to perform qr decomp on the vectors to ensure orthogonality when eigenvalues are very close


def projUNN_D(A, a, b, project_on=True):
    rank = a.shape[-1]
    a_hat = conjugate_transpose(A) @ a
    A_tilde = A + add_outer_products(a, b)

    a_and_b = torch.cat((b, a_hat), dim=-1)
    a_and_b, _ = torch.linalg.qr(a_and_b, mode="reduced")
    projectors = conjugate_transpose(a_and_b)

    c_ij = torch.reshape(conjugate_transpose(a) @ a, (rank, rank))
    a_obasis = projectors @ a_hat
    b_obasis = projectors @ b

    sub_arr = add_outer_products(a_obasis, b_obasis)
    sub_arr += add_outer_products(b_obasis, a_obasis)
    sub_arr += torch.einsum("an,bm,nm->ab", b_obasis, torch.conj(b_obasis), c_ij)
    s, D = torch.linalg.eig(sub_arr.type(torch.double))
    s = (1.0 / torch.sqrt(s + 1) - 1).type(A.dtype)
    u = a_and_b.type(D.dtype) @ D
    if project_on:
        return A_tilde + add_outer_products(A_tilde.type(D.dtype) @ (s * u), u).type(
            A.dtype
        )
    else:
        return add_outer_products(A_tilde.type(D.dtype) @ (s * u), u).type(
            A.dtype
        ) + add_outer_products(a, b)


def projUNN_T(A, a, b, project_on=True):
    if len(A.shape) == 3:
        batched = True
    else:
        batched = False
    a_hat = torch.matmul(conjugate_transpose(A), a)

    a_and_b = torch.cat((b, a_hat), dim=-1)
    a_and_b, _ = torch.linalg.qr(a_and_b, mode="reduced")
    projectors = conjugate_transpose(a_and_b)

    a_obasis = projectors @ a_hat
    b_obasis = projectors @ b

    sub_arr = 0.5 * add_outer_products(a_obasis, b_obasis)
    sub_arr -= 0.5 * add_outer_products(b_obasis, a_obasis)
    if sub_arr.shape[-1] == 2:
        s, D = dim2_eig(sub_arr)
    else:
        s, D = torch.linalg.eigh(sub_arr)

    s = torch.exp(s) - 1.0

    if batched:
        s = s.unsqueeze(1)

    if torch.is_complex(A):
        u = a_and_b @ D
        if project_on:
            return A + add_outer_products(A.type(D.dtype) @ (s * u), u).type(A.dtype)
        else:
            return add_outer_products(A.type(D.dtype) @ (s * u), u).type(A.dtype)
    else:
        u = a_and_b.type(D.dtype) @ D
        if project_on:
            return A + add_outer_products(A.type(D.dtype) @ (s * u), u).type(A.dtype)
        else:
            return add_outer_products(A.type(D.dtype) @ (s * u), u).type(A.dtype)


class OrthoRegularizer:
    def __init__(self, net, kernel_size, penalize_support=False, dtype="real"):
        self.get_params(net)
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        self.penalize_support = penalize_support
        if penalize_support:
            self.mult_term = 1.0
        else:
            self.mult_term = -1.0
        if dtype == "real":
            self.fft_op = torch.fft.rfft2
        else:
            self.fft_op = torch.fft.fft2

        self.get_regularizing_locs()
        self.get_projectors()

    def get_params(self, net):
        self.params = []
        self.sizes = []
        for param in net.parameters():
            if len(param.shape) == 3:
                self.params.append(
                    {"param": param, "size": param.ortho_regularizer_size}
                )
        if len(self.params) == 0:
            raise ValueError(
                "no orthogonal convolutional parameters found to regularize"
            )

    def get_regularizing_locs(self):
        for param_dict in self.params:
            param_dict["locs"] = self.kernel_to_locs(
                self.kernel_size, param_dict["size"]
            )

    def kernel_to_locs(self, kernel_size, input_size):
        locs = []
        for i in range(
            kernel_size[0] // 2 - kernel_size[0] + 1,
            kernel_size[0] - kernel_size[0] // 2,
        ):
            for j in range(
                kernel_size[1] // 2 - kernel_size[1] + 1,
                kernel_size[1] - kernel_size[1] // 2,
            ):
                locs.append([i, j])
        return locs

    def get_projectors(self):
        for param_dict in self.params:
            self.convert_to_projector(param_dict)

    def convert_to_projector(self, param_dict):
        basis = torch.zeros(
            (len(param_dict["locs"]), *param_dict["size"]),
            device="cuda",
        )
        locs = np.stack(param_dict["locs"])
        locs = np.concatenate([np.arange(len(locs))[:, None], locs], 1)
        locs = torch.Tensor(locs).long()
        basis[locs] = 1.0
        # for i, loc in enumerate(param_dict["locs"]):
        #     print(loc,basis[i].shape)
        #     basis[i,loc[0],loc[1]] = 1.0
        param_dict["projector"] = (
            self.fft_op(basis, s=param_dict["size"], norm="ortho").flatten(1).conj()
        )

    def regularize(self):
        loss = 0.0
        for param_dict in self.params:
            loss += self.mult_term * self.regularize_term(
                param_dict["projector"], param_dict["param"]
            )
        return loss

    def regularize_term(self, projector, param):
        # print(projector.shape,param.shape)
        # asdf
        temp = torch.tensordot(
            projector,
            param,
            dims=([-1], [0]),
        )
        return torch.sum(torch.abs(temp) ** 2)
