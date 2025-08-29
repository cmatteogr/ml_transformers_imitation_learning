"""
To understand how Transformers work
https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
https://poloclub.github.io/transformer-explainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
import math

class Residual(nn.Module):
    """
    Residual layer (ResNet), provides stability
    Helps in training deeper networks by minimizing gradient issues
    https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning
    """
    def __init__(self, fn):
        super().__init__()
        # get the function as input, the input is the model until that point
        self.fn = fn

    def forward(self, x, **kwargs):
        # use the function (model) and sum the function output and the input
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Normalization Layer, Stabilizes training, improves convergence
    Read more about Normalization layers: https://theaisummer.com/normalization
    """
    def __init__(self, dim, fn):
        super().__init__()
        # init normalization layer, define dimension
        self.norm = nn.LayerNorm(dim)
        # init the function, the model until that point
        self.fn = fn

    def forward(self, x, **kwargs):
        # apply normalization and the function
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    This is the MLP module (Multilayer Perceptron), where we enrich the multi-attention head outputs
    and apply RELU to filter attention outputs. Transforms the attention outputs, adding complexity.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Linear Layer to expand and enrich attention outputs
        # ReLU to filter the relevant
        # Dropout to prevent overfitting
        # Linear Layer to compress back the output to the original dimension
        # Dropout to prevent overfitting
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # execute the MLP module
        return self.net(x)

class PositionalEncoding(nn.Module):
    """
    For each event in the sequence define the position and generates and embedding representation useful
    to add to the input and provide more information to the Transformer, Provides sequence order context to the model.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # init Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # vector full of zeros to work as a placeholder for the positional encoding items
        pe = torch.zeros(max_len, d_model)
        # position vector based on the max_len (the sequence length). [0,1,2,3...n]
        position = torch.arange(0, max_len).unsqueeze(1)
        #
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Attention(nn.Module):
    """
    Multi-Attention Layer head, captures the dependencies or relations in the sequence,
    """
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        # define the number of heads
        self.heads = heads
        # define the dimension of each head
        self.scale = dim ** -0.5

        # init the layer (matrix) for the Queries, Keys, and Values. That's why the output is dim*3
        # In this case we are not using bias, not completely sure why, TO REVIEW.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        # define the output layer, after apply attention, include Dropout
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """#
        * b = batch_size
        * n = sequence_length
        * h = attention_heads

        * q = queries
        * k = keys
        * v = values
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Calculate the dot product, using the queries and keys, Apply normalization based on dimension using self.scale
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # define if use mask to avoid data leakage, and use self-attention
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        # apply softmax for the scaled dot queries and keys products, to get attention scores normalized
        self.attn = dots.softmax(dim=-1)

        # Using Einstein notation apply attention multiplication with values
        # https://docs.pytorch.org/docs/stable/generated/torch.einsum.html
        out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
        # Change the position of the result to have the expected output
        out = rearrange(out, 'b h n d -> b n (h d)')
        # apply the output layer
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        # create a list of module, it's the sequence of Attention-Heads
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # add Attention-Heads, Attention -> PreNorm -> Residual
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # for each layer apply attention and feed forward
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, n_channel, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        """

        :param n_channel: n of features, for each time step. This inputs are projected to a internal Transformer space dim, to apply the same split by words in the LLMs where each word-embedding has keys, queries and values matrices related
        :param len_sw: length of the sliding window = sequence length = time steps = number of frames
        :param n_classes: output classes, for classification. In LLMs it's all the possible words, here it's to possible person status: running, walking, etc.
        :param dim: Internal transformer dimension, input data is projected to a space to this dimension
        :param depth: Number of transformer layers, stack of layers transformers
        :param heads: Number of attention heads
        :param mlp_dim: Multilayer percepton layer MLP, to connect with the full neural network
        :param dropout: Dropout rate
        """
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.classifier = nn.Linear(dim, n_classes)


    def forward(self, forward_seq):
        """
        Put all together
        """
        x = self.patch_to_embedding(forward_seq)
        x = self.position(x)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class _Seq_Transformer(nn.Module):
    # the transformer used in TS-TCC: without positional encoding
    def __init__(self, patch_size, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_size, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t