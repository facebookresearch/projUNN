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


def LSI_approximation(A, k, eps=1e-6):
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


def projUNN_D(A, a, b, stable_adder = 1e-8, noise_adder = 0, project_on=True):
    if len(A.shape) == 3:
        batched = True
        batch_size = A.shape[0]
    else:
        batched = False

    if noise_adder:
        a += noise_adder*a.new(a.size()).normal_(0,1)
        b += noise_adder*b.new(b.size()).normal_(0,1)

    rank = a.shape[-1]
    a_hat = conjugate_transpose(A) @ a
    A_tilde = A + add_outer_products(a, b)

    a_and_b = torch.cat((b, a_hat), dim=-1)
    if batched:
        a_and_b = fast_batch_qr_modified(a_and_b)
    else:
        a_and_b, _ = torch.linalg.qr(a_and_b, mode = 'reduced')             # native torch operation seems to be fastest
    projectors = conjugate_transpose(a_and_b)

    if batched:
        c_ij =  torch.reshape(conjugate_transpose(a)@a, (batch_size,rank, rank))
    else:
        c_ij =  torch.reshape(conjugate_transpose(a)@a, (rank, rank))
    a_obasis = projectors @ a_hat
    b_obasis = projectors @ b

    sub_arr = add_outer_products(a_obasis, b_obasis)
    sub_arr += add_outer_products(b_obasis, a_obasis)
    sub_arr += torch.einsum("an,bm,nm->ab", b_obasis, torch.conj(b_obasis), c_ij)
    
    if sub_arr.shape[-1] == 2:
        s, D = dim2_eig(sub_arr)
    else:
        s, D = torch.linalg.eig(sub_arr)

    if batched:
        s = s.unsqueeze(1)

    s = (1.0 / torch.sqrt(s + 1 + stable_adder) - 1).type(A.dtype)
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
    if batched:
        a_and_b = fast_batch_qr_modified(a_and_b)
    else:
        a_and_b, _ = torch.linalg.qr(a_and_b, mode = 'reduced')             # native torch operation seems to be fastest
    projectors = conjugate_transpose(a_and_b)

    a_obasis = projectors @ a_hat
    b_obasis = projectors @ b

    sub_arr = 0.5 * add_outer_products(a_obasis, b_obasis)
    sub_arr -= 0.5 * add_outer_products(b_obasis, a_obasis)
    if sub_arr.shape[-1] == 2:
        s, D = dim2_eig(sub_arr)
    else:
        s, D = torch.linalg.eig(sub_arr)

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


def fast_batch_qr(A):
    # Uses regular Gram-Schmidt and is much faster over batches
    # may be unstable when matrix is not skinny
    def proj(M, v):
        return M@(conjugate_transpose(M)@v)
    def norm_squared(M):
        return torch.sum(M*torch.conj(M),dim = -2,keepdims = True)
    A[:,:,:1] = A[:,:,:1] / torch.sqrt(norm_squared(A[:,:,:1]))
    for i in range(1,A.shape[-1]):
        A[:,:,i:i+1] = A[:,:,i:i+1] - proj(A[:,:,:i],A[:,:,i:i+1])
        A[:,:,i:i+1] = A[:,:,i:i+1] / torch.sqrt(norm_squared(A[:,:,i:i+1]))
    return A

def fast_batch_qr_modified(A):
    # slightly slower but a bit more stable
    def proj(M, v):
        return M@(conjugate_transpose(M)@v)
    def norm_squared(M):
        return torch.sum(M*torch.conj(M),dim = -2,keepdims = True)
    A[:,:,:1] = A[:,:,:1] / torch.sqrt(norm_squared(A[:,:,:1]))
    for i in range(1,A.shape[-1]):
        A[:,:,i:] = A[:,:,i:] - proj(A[:,:,i-1:i],A[:,:,i:])
        A[:,:,i:] = A[:,:,i:] / torch.sqrt(norm_squared(A[:,:,i:]))
    return A


def one_to_two_fft_indices(i, kernel_size):
    return i // kernel_size[1], i % kernel_size[1]

 

def two_to_one_fft_indices(i,j, kernel_size):
    return i*kernel_size[1] + j

 

def get_matching_hermitian_fft_dim(i, kernel_size, input_size = None):
    if input_size is None:
        input_size = kernel_size.copy()
        input_size[1] *= 2
    a,b = one_to_two_fft_indices(i, kernel_size)
    if b == 0 or (b == (kernel_size[1]-1) and (input_size[1]%2) == 0):
        if a == 0:
            return -1
        elif a*2 == (kernel_size[0]):
            return -1
        else:
            return two_to_one_fft_indices( -1*a % kernel_size[0],b, kernel_size )
    else:
        return -1

 

def is_fft_dim_double(i, kernel_size, input_size = None):
    if input_size is None:
        input_size = kernel_size.copy()
        input_size[1] *= 2
    a,b = one_to_two_fft_indices(i, kernel_size)
    if b == 0 or (b == (kernel_size[1]-1) and (input_size[1]%2) == 0):
        return True
    else:
        return False

class OrthoRegularizer:
    def __init__(self, net, kernel_size, device = 'cuda:0', penalize_support = False, use_orthogonal = True):
        self.get_params(net)
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size,kernel_size]
        else:
            self.kernel_size = kernel_size
        self.penalize_support = penalize_support
        if penalize_support:
            self.mult_term = 1.
        else:
            self.mult_term = -1.
        if use_orthogonal:
            self.fft_op = torch.fft.rfft2
            self.dtype = torch.float32
        else:
            self.fft_op = torch.fft.fft2
            self.dtype = torch.complex64
        self.device = device
        self.use_orthogonal = use_orthogonal

        self.get_regularizing_locs()
        self.get_projectors()



    def get_params(self, net):
        self.params = []
        self.sizes = []
        for param in net.parameters():
            if hasattr(param, 'ortho_regularizer_size'):
                self.params.append({'param': param, 'size': param.ortho_regularizer_size})
        if len(self.params) == 0:
            raise ValueError('no orthogonal convolutional parameters found to regularize')

    def get_regularizing_locs(self):
        for param_dict in self.params:
            param_dict['locs'] = self.kernel_to_locs(self.kernel_size,param_dict['size'])

    def kernel_to_locs(self, kernel_size, input_size):
        locs = []
        for i in range(kernel_size[0]//2-kernel_size[0]+1,kernel_size[0] - kernel_size[0]//2 ):
            for j in range(kernel_size[1]//2-kernel_size[1]+1,kernel_size[1] - kernel_size[1]//2):
                locs.append([i,j])
        return locs

    def construct_mask(self, kernel_size,input_size):
        mask = torch.zeros(kernel_size, dtype = default_complex_dtype).to(self.device)
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                k = two_to_one_fft_indices( i,j, kernel_size )
                match = is_fft_dim_double(k, kernel_size, input_size)
                if match:
                    mask[i,j] = 1.
        return mask

    def get_projectors(self):
        for param_dict in self.params:
            self.convert_to_projector(param_dict)

    def convert_to_projector(self, param_dict):
        basis = torch.zeros(  (len(param_dict['locs']), param_dict['size'][0],param_dict['size'][1]) ,
                                dtype = self.dtype, device = self.device)
        for i, loc in enumerate(param_dict['locs']):
            basis[i][loc[0]][loc[1]] = 1.
        param_dict['projector'] = self.fft_op(basis, s=param_dict['size'], norm = 'ortho')
        if self.use_orthogonal:
            param_dict['mask'] = self.construct_mask(param_dict['projector'].shape[1:],param_dict['size']).reshape(-1)
        param_dict['projector'] = param_dict['projector'].reshape(len(param_dict['locs']), -1).conj()

    def regularize(self):
        loss = 0.
        for param_dict in self.params:
            loss += self.mult_term*self.regularize_term(param_dict)
        return loss

    def regularize_term(self, param_dict):
        if self.use_orthogonal:
            temp = torch.tensordot(param_dict['projector']*param_dict['mask'],param_dict['param'], dims = ([-1], [0]))
            temp += torch.tensordot(param_dict['projector']*(1-param_dict['mask']),param_dict['param'], dims = ([-1], [0]))
            temp += torch.tensordot(torch.conj(param_dict['projector'])*(1-param_dict['mask']),torch.conj(param_dict['param']), dims = ([-1], [0]))
        else:
            temp = torch.tensordot(param_dict['projector'],param_dict['param'], dims = ([-1], [0]))            
        return torch.sum(torch.abs(temp)**2)

    def project_grad(self):
        for param_dict in self.params:
            param_dict['param'].grad = self.project_term(param_dict)


    def project_term(self,param_dict):
        if self.use_orthogonal:
            temp = torch.tensordot(param_dict['projector']*param_dict['mask'],param_dict['param'].grad, dims = ([-1], [0]))
            temp += torch.tensordot(param_dict['projector']*(1-param_dict['mask']),param_dict['param'].grad, dims = ([-1], [0]))
            temp += torch.tensordot(torch.conj(param_dict['projector'])*(1-param_dict['mask']),torch.conj(param_dict['param'].grad), dims = ([-1], [0]))
        else:
            temp = torch.tensordot(param_dict['projector'],param_dict['param'].grad, dims = ([-1], [0]))
        return torch.einsum('abc,ad->dbc',temp, torch.conj(param_dict['projector']))

