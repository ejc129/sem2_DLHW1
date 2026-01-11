from pathlib import Path

import torch
#from torch._C import float32

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
import torch.nn.functional as F
import torch.nn as nn

class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        super().__init__(in_features= in_features, out_features= out_features, bias= bias)
        with torch.no_grad():
          # nn.Parameter wrap so we tell torch that this is a parameter
            # to be learned/optimized through training
          # .detach() to return a Tensor that shares the same storage as 
            # self.weight but has no Autograd history. Ensures a clean new leaf
          # this self.bias is here because we have nn.Linear parent
          self.weight = nn.Parameter(
            self.weight.detach().to(torch.float16), requires_grad= False
          )
          # this self.weight is here because we have nn.Linear parent
          if self.bias is not None:
            self.bias = nn.Parameter(
              self.bias.detach().to(torch.float16), requires_grad= False
            )
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        # TODO: Implement me
        if x.dtype != torch.float32:
          x = x.to(dtype= torch.float32)

        # we cast to fp16
        x_half = x.to(dtype= torch.float16)
        weight_half = self.weight.to(dtype= torch.float16, device= x.device)
        
        if self.bias is not None:
          bias_half = self.bias.to(dtype= torch.float16, device= x.device)
        else: 
          bias_half = None
        
        # functions as a stateless nn.Linear, so we don't have to store the weight and bias
        # saves on the memory that is so precious <3 
        y_half = F.linear(x_half, weight_half, bias_half)
        y = y_half.to(dtype= torch.float32)
        return y

class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )
        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
