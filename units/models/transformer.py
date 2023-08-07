""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import nn


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, n_head, d_head, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head

        d_proj = n_head * d_head

        self.qkv = nn.Linear(d_in, d_proj * 3)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_proj, d_in)

        self.apply(init_weights)

    def forward(self, input, mask=None, memory=None):
        batch, length, dim = input.shape

        qkv = self.qkv(input)
        q, k, v = qkv.chunk(3, dim=-1)

        # memory remember last layer's attention key and value
        # These behaviour can be effective to remember long sequence.
        with torch.no_grad():
            if memory is not None and "k" in memory:
                k = torch.cat((memory["k"], k), 1)
                v = torch.cat((memory["v"], v), 1)
            next_memory = {"k": k, "v": v}

        q = q.reshape(batch, -1, self.n_head, self.d_head)
        k = k.reshape(batch, -1, self.n_head, self.d_head)
        v = v.reshape(batch, -1, self.n_head, self.d_head)

        attn = (q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)) / (self.d_head ** 0.5)

        if mask is not None:
            # TODO: use of different value to enable fp16
            attn.masked_fill_(mask, float("-inf"))

        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)

        out = attn @ v.permute(0, 2, 1, 3)
        out = out.transpose(1, 2).reshape(batch, length, dim)
        out = self.out(out)

        return out, next_memory


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_in, d_kv, n_head, d_head, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head

        d_proj = n_head * d_head

        self.q = nn.Linear(d_in, d_proj)
        self.kv = nn.Linear(d_kv, d_proj * 2)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_proj, d_in)

        self.apply(init_weights)

    def forward(self, q, kv, mask=None, memory=None):
        batch, length, dim = q.shape

        q = self.q(q)

        # memory remember last layer's attention key and value
        # These behaviour can be effective to remember long sequence.
        if memory is not None:
            with torch.no_grad():
                k = memory["k"]
                v = memory["v"]
                mask = memory["mask"]

                next_memory = memory

        else:
            kv = self.kv(kv)
            k, v = kv.chunk(2, dim=-1)
            k = k.reshape(batch, -1, self.n_head, self.d_head)
            v = v.reshape(batch, -1, self.n_head, self.d_head)

            next_memory = {"k": k, "v": v, "mask": mask}

        q = q.reshape(batch, -1, self.n_head, self.d_head)

        attn = (q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)) / (self.d_head ** 0.5)

        if mask is not None:
            # TODO: use of different value to enable fp16
            attn.masked_fill_(mask, float("-inf"))

        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)

        out = attn @ v.permute(0, 2, 1, 3)
        out = out.transpose(1, 2).reshape(batch, length, dim)
        out = self.out(out)

        return out, next_memory


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class FeedForwardLayer(nn.Module):
    def __init__(self, dim, dim_mlp, act_layer=nn.GELU, drop_units=0.1):
        super().__init__()
        self.dense1 = nn.Linear(dim, dim_mlp)
        self.act = act_layer()
        self.dropout = nn.Dropout(drop_units)
        self.dense2 = nn.Linear(dim_mlp, dim)

        self.apply(init_weights)

    def forward(self, x):
        return self.dense2(self.dropout(self.act(self.dense1(x))))


class MLP(nn.Module):
    def __init__(self, num_layers, dim, mlp_ratio, drop_path=0.1, drop_units=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList(
            [
                FeedForwardLayer(dim, dim * mlp_ratio, drop_units=drop_units)
                for _ in range(num_layers)
            ]
        )
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)]
        )
        self.droppath = DropPath(drop_path)

        self.apply(init_weights)

    def forward(self, x):
        for i in range(self.num_layers):
            x_residual = self.mlp_layers[i](self.layernorms[i](x))
            x = x + self.droppath(x_residual)
        return x


class MultiwayFeedForwardLayer(nn.Module):
    def __init__(self, n_experts, dim, dim_mlp, act_layer=nn.GELU, drop_units=0.1):
        super().__init__()

        self.dim = dim
        self.ff = nn.ModuleList(
            [
                FeedForwardLayer(dim, dim_mlp, act_layer, drop_units)
                for _ in range(n_experts)
            ]
        )

    def compute_expert_idx(self, expert_idx):
        bs, seq_len = expert_idx.shape[:2]
        feature_dim_idx = (
            torch.arange(0, self.dim, device=expert_idx.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        feature_dim_idx = feature_dim_idx.repeat(bs, seq_len, 1)  # bs x seq_len x dim
        expert_idx = expert_idx.unsqueeze(2) * self.dim  # bs x seq_len x dim

        return expert_idx + feature_dim_idx

    def forward(self, x, expert_idx):
        feat_all = torch.cat(
            [ff(x) for ff in self.ff], dim=-1
        )  # bs x seq_len x (dim * 2)
        expert_idx = self.compute_expert_idx(expert_idx)  # bs x seq_len x dim
        return torch.gather(feat_all, 2, expert_idx)


class MultiwayMLP(nn.Module):
    def __init__(
        self, num_experts, num_layers, dim, mlp_ratio, drop_path=0.1, drop_units=0.0
    ):
        super().__init__()

        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList(
            [
                MultiwayFeedForwardLayer(
                    num_experts, dim, dim * mlp_ratio, drop_units=drop_units
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)]
        )
        self.droppath = DropPath(drop_path)

        self.apply(init_weights)

    def forward(self, x, expert_idx):
        for i in range(self.num_layers):
            x_residual = self.mlp_layers[i](self.layernorms[i](x), expert_idx)
            x = x + self.droppath(x_residual)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, dim, n_head, mlp_ratio=4, drop_path=0.1, drop_units=0.1, drop_attn=0.0
    ):
        """
        Args:
            dim (int): hidden dimension
            n_head (int): number of attention heads
            mlp_ratio (int): hidden dim expansion in mlp
            drop_path (float): droppath ratio for both attention and feedforward
            drop_units (float): dropout ratio for feedforward
            drop_attn (float): dropout ratio for attention
        """
        super().__init__()

        self.dim = dim

        self.self_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            dim, n_head, dim // n_head, dropout=drop_attn
        )

        self.droppath = DropPath(drop_path)  # drop path for attn

        self.mlp = MLP(1, dim, mlp_ratio, drop_path=drop_path, drop_units=drop_units)

    def forward(self, input, mask=None):
        out = self.self_norm(input)
        out, _ = self.self_attn(out, mask)
        input = input + self.droppath(out)

        out = self.mlp(input)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        n_experts=1,
        mlp_ratio=4,
        drop_path=0.1,
        drop_units=0.1,
        drop_attn=0.0,
        self_attention=True,
        cross_attention=True,
    ):
        """
        Args:
            dim (int): hidden dimension
            n_head (int): number of attention heads
            mlp_ratio (int): hidden dim expansion in mlp
            drop_path (float): droppath ratio for both attention and feedforward
            drop_units (float): dropout ratio for feedforward
            drop_attn (float): dropout ratio for attention
        """
        super().__init__()

        self.self_attention = self_attention
        if self_attention:
            self.self_norm = nn.LayerNorm(dim, eps=1e-6)
            self.self_attn = MultiHeadAttention(
                dim, n_head, dim // n_head, dropout=drop_attn
            )

        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
            self.cross_attn = MultiHeadCrossAttention(
                dim,
                dim,
                n_head,
                dim // n_head,
                dropout=drop_attn,
            )

        self.dim = dim
        self.n_experts = n_experts
        if n_experts > 1:
            self.mlp = MultiwayMLP(
                n_experts, 1, dim, mlp_ratio, drop_path=drop_path, drop_units=drop_units
            )
        else:
            self.mlp = MLP(
                1, dim, mlp_ratio, drop_path=drop_path, drop_units=drop_units
            )
        self.droppath = DropPath(drop_path)

    def forward(
        self,
        input,
        source,
        source_mask,
        memory=None,
        mask=None,
        expert_idx=None,
    ):
        """
        Args:
            input (Tensor): Input query with shape (n_query, B, embed_dims)
            source (Tensor): Encoder output. (8, ?, dim)
            source_mask (Tensor[bool]): (B, ?)
            memory (list[dict]): memory of attn "key" and "value"
            mask (Tensor): mask tensor. Useful when we use
                "teacher forcing" for training autoregressive model.

        Returns:
            out
            next_memory
        """
        if memory is None:
            memory = (None, None)

        if self.self_attention:
            out = self.self_norm(input)
            out, next_memory1 = self.self_attn(
                out,
                memory=memory[0],
                mask=mask,
            )
            input = input + self.droppath(out)

        if self.cross_attention:
            out = self.cross_norm(input)
            out, next_memory2 = self.cross_attn(
                out,
                source,
                source_mask,
                memory=memory[1],
            )
            input = input + self.droppath(out)

        if self.n_experts > 1:
            out = self.mlp(input, expert_idx)
        else:
            out = self.mlp(input)

        return out, (next_memory1, next_memory2)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x


def autoregressive_mask(query_size, memory_size, device):
    mask = torch.triu(
        torch.ones(
            query_size, query_size + memory_size, device=device, dtype=torch.bool
        ),
        diagonal=memory_size + 1,
    )

    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layers, autoregressive=True):
        """
        Args:
            decoder_layers (List[TransformerDecoderLayer]): decoder layers
            autoregressive (bool): autoregressive or not
        """

        super().__init__()

        self.layers = nn.ModuleList(decoder_layers)
        self.norm = nn.LayerNorm(self.layers[0].dim, eps=1e-6)

        self.autoregressive = autoregressive

    def forward(
        self,
        input,
        pos,
        source,
        source_mask,
        memory=None,
        expert_idx=None,
    ):
        """
        Args:
            input (Tensor): Input query with shape (n_query, B, embed_dims)
            pos (Tensor): (B, n_query_pos, embed_dims)
            source (Tensor): Encoder output. (8, ?, dim)
            source_mask (Tensor[bool]): (B, ?)
            memory (List[dict]): [Attention key,value] Memory from last decoder block.

        Returns:
            intermediate (List[Tensor]): list of intermediate tensors (B, n_query, dim)
            memories (List[dict]): list of attention key-query memory
        """
        out = input
        intermediate = []
        memories = []
        memory_size = 0 if memory is None else memory[0][0]["k"].shape[-2]

        out = out if pos is None else out + pos

        mask = None
        if self.autoregressive:
            mask = autoregressive_mask(input.shape[1], memory_size, device=input.device)

        for i, layer in enumerate(self.layers):
            cur_memory = None
            if memory is not None:
                cur_memory = memory[i]

            out, next_memory = layer(
                out,
                source,
                source_mask,
                memory=cur_memory,
                mask=mask,
                expert_idx=expert_idx,
            )
            memories.append(next_memory)

            out_norm = out
            if i == len(self.layers) - 1:
                out_norm = self.norm(out_norm)

            intermediate.append(out_norm)

        return intermediate, memories
