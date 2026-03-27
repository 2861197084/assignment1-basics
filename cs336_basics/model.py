import torch
import torch.nn as nn


class Linear(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i,oi->...o', x, self.weight) + (
            self.bias if self.bias is not None else 0
        )


class Embedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    weight: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings,embedding_dim), **kwargs)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    d_model: int
    eps: float
    weight: torch.Tensor

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, **kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_out = x * var.rsqrt() * self.weight
        return x_out.to(in_dtype)

def norm(x:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """Functional RMS normalization without learnable parameters"""
    var = x.pow(2).mean(dim=-1, keepdim=True) + eps
    return x * var.rsqrt()

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    d_model: int
    d_ff: int

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    

class RotaryPositionalEmbedding(nn.Module):
    theta: float
    d_k: int
    max_seq_len: int
    device: torch.device | None

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass