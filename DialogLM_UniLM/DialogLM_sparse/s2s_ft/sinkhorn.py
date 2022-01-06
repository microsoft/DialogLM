import torch
import torch.nn as nn
from fairseq.modules import (
MultiheadAttention,
)
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from torch.autograd.function import Function
from functools import partial, wraps, reduce
from operator import mul
from inspect import isfunction
import torch.nn.functional as F
import logging

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

class SinkhornSelfAttention(MultiheadAttention):
    def __init__(self,
        embed_dim=768,
        num_heads=12,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=True,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        sinkhorn_iter=7,
        bucket_size=256,
        temperature=0.7
        ):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout,
                         bias, add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)


        self.bucket_size = bucket_size
        logging.info('bucket size: {}'.format(bucket_size))
        self.sinkhorn_iter = sinkhorn_iter

        self.sort_net = AttentionSortNet(self.num_heads, bucket_size, bucket_size, self.head_dim, temperature, sinkhorn_iter)

        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(
        self,
        query,
        # key: Optional[Tensor],
        # value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        query = query.transpose(0, 1)
        tgt_len, bsz, emb_dim = query.size()
        bh = bsz * self.num_heads
        q = self.q_proj(query).permute(1, 0, 2) # B * T * D

        k = self.k_proj(query).permute(1, 0, 2)
        v = self.v_proj(query).permute(1, 0, 2)

        qkv = (q, k, v)
        merge_heads_fn = partial(merge_heads, self.num_heads)
        q, k, v = map(merge_heads_fn, qkv) # B * H * T * D

        buckets = q.shape[2] // self.bucket_size
        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v)) # BH * T * D
        
        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, buckets), (k, v)) # BH * bct * n_b * D

        R = self.sort_net(q, k)
        R = R.type_as(q).to(q)

        b_k_r = reorder_buckets(b_k, R).reshape(bh, buckets, -1, self.head_dim) # BH * bct * 2n_b * D
        b_v_r = reorder_buckets(b_v, R).reshape(bh, buckets, -1, self.head_dim) # BH * bct * 2n_b * D

        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * self.scaling

        mask_value = -10000

        # mask
        if key_padding_mask is not None:
            q_mask = default(key_padding_mask.eq(0), lambda: torch.ones((bsz, tgt_len), device=q.device).bool())
            kv_mask = q_mask
            mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask) # B * bkt * n_b
            expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, self.num_heads))
            mq, mk = map(expand_head_and_merge_into_batch, (mq, mk)) # BH * bkt * n_b
            mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))
            mk_r = mk_r.reshape(bh, buckets, -1)
            mk = torch.cat((mk_r, mk), dim=2)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask
        dots = dots.softmax(dim=-1)
        dots = self.dropout_module(dots)

        attn = torch.einsum('buij,buje->buie', dots, b_v)
        attn = unbucket(attn)
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, emb_dim)
        # attn = self.out_proj(attn)
        # attn = attn.transpose(0, 1)
        attn = attn.contiguous().view(bsz, tgt_len, emb_dim)
        attn_weights: Optional[Tensor] = None
        
        return attn, attn_weights


class AttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, kv_bucket_size, dim, temperature, sinkhorn_iter, n_sortcut = 0):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = kv_bucket_size
        self.dim = dim
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

    def forward(self, q, k, topk=1):
        bh, *_, bucket_size, kv_bucket_size, device, dtype, dim = *q.shape, self.bucket_size, self.kv_bucket_size, q.device, q.dtype, self.dim
        b = bh // self.heads

        buckets = q.shape[1] // bucket_size
        kv_buckets = k.shape[1] // kv_bucket_size

        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        b_k = bucket(kv_buckets, k)

        sq = b_q.mean(dim=2)
        sk = b_k.mean(dim=2)

        R = torch.einsum('bie,bje->bij', sq, sk).to(q) * (dim ** -0.5)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)
