# _*_coding:utf-8_*_
# created by Amuu on 2023/7/26

import torch
from torch import nn
import math


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(dim, dtype=torch.float) / dim))
    sinusoid_inp = torch.einsum("i,j->ij", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def duplicate_interleave(m):
    dim0 = m.shape[0]
    m = m.view(-1, 1).repeat(1, 2).view(dim0, -1)
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin = duplicate_interleave(sin * scale)
    cos = duplicate_interleave(cos * scale)
    return (x * cos) + (rotate_every_two(x) * sin)


def swish(x):
    return x * torch.sigmoid(x)


class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        return self._forward_impl(x, offset, downscale)

    def forward_reverse(self, x, offset=0, downscale=False):
        return self._forward_impl(x, -offset, downscale)

    def _forward_impl(self, x, offset, downscale):
        length = x.shape[2]
        min_pos = -(length + offset) // 2 if offset < 0 else 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, (-sin if offset < 0 else sin), cos, scale)
        return x


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma):
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)

        self.xpos = XPOS(hidden_size)

    def forward(self, X):
        sequence_length = X.shape[2]
        D = self._get_D(sequence_length).to('cuda:0')

        Q = X @ self.W_Q
        K = X @ self.W_K

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        att = (Q @ K.permute(0, 1, 3, 2)) * D.unsqueeze(0).unsqueeze(0)

        return att @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        Q = x_n @ self.W_Q
        K = x_n @ self.W_K

        Q = self.xpos(Q, n + 1)
        K = self.xpos(K, n + 1, downscale=True)

        V = x_n @ self.W_V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return Q @ s_n, s_n

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0

        return D


class SimpleRetentions(nn.Module):
    def __init__(self, hidden_size, gamma):
        super(SimpleRetentions, self).__init__()
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.W_R = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)

    def forward(self, X):
        return X + self.gamma * torch.tanh(X @ self.W_R)


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads):
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads

        self.gammas = (
                1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))
        ).detach().cpu().tolist()

        self.W_G = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, hidden_size)

        self.retentions = nn.ModuleList([
            SimpleRetentions(self.head_size, gamma) for gamma in self.gammas
        ])

    def forward(self, X):
        Y = [retention(X[:, :, :, i * self.head_size:(i + 1) * self.head_size]) for i, retention in
             enumerate(self.retentions)]
        Y = torch.cat(Y, dim=3)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(X.shape)

        return (torch.sigmoid(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        Y = []
        s_ns = []
        for i, retention in enumerate(self.retentions):
            y, s_n = retention.forward_recurrent(
                x_n[:, :, i * self.head_size:(i + 1) * self.head_size], s_n_1s[i], n
            )
            Y.append(y)
            s_ns.append(s_n)

        Y = torch.cat(Y, dim=1)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(x_n.shape)

        return (torch.sigmoid(x_n @ self.W_G) * Y) @ self.W_O, s_ns


class RetNet(nn.Module):
    def __init__(self, node_number, history_length, prediction_length, hidden_dim, ffn_size, heads, layers):
        super(RetNet, self).__init__()
        self.node_number = node_number
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.layers = layers

        self.retentions = nn.ModuleList([MultiScaleRetention(hidden_dim, heads) for _ in range(layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.SiLU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.conv = nn.Conv2d(1, hidden_dim, (1, 1))
        self.conv1 = nn.Conv2d(hidden_dim, 1, (1, 1))
        self.conv2 = nn.Conv2d(history_length, prediction_length, (1, 1))

    def forward(self, X, device):
        X = X.to(device)
        X = self.conv(X.permute(0, 3, 2, 1)).perm(0, 3, 2, 1)

        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        X = self.conv1(X.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        X = self.conv2(X.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return X.squeeze(3)
