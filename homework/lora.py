from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear
from .low_precision import Linear4Bit

import torch.nn as nn

class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # Freeze the base linear layer (inherited from HalfLinear)
        for param in self.parameters():
            param.requires_grad = False

        # Initialize LoRA layers in float32
        # where A is (in_features x lora_dim) and B is (lora_dim x out_features)
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)


        # Initialize LoRA weights
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_b.weight)



        # Ensure LoRA layers are trainable and in float32...
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original dtype
        original_dtype = x.dtype
        # cmpute base linear layer output using parent's forward (handles dtype conversion)
        base_output = super().forward(x)



        # Compute LoRA adaptation in float32
        x_float32 = x.to(torch.float32)
        lora_output = self.lora_b(self.lora_a(x_float32))
        # convert LoRA output to original dtype and add to base output
        lora_output = lora_output.to(original_dtype)
        output = base_output + lora_output
        # yay!

        return output
class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        group_size = 16
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)



def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    
    return net

