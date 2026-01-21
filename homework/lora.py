from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear

import torch.nn as nn


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        alpha: float,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        rank = lora_dim
        super().__init__(in_features, out_features, bias)

        self.weight.requires_grad = False
        if self.bias is not None: self.bias.requires_grad = False 

        self.lora_a = nn.Linear(in_features, rank, bias = False).to(torch.float32)
        self.lora_b = nn.Linear(rank, out_features, bias = False).to(torch.float32)

        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)

        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha/rank

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the base HalfLinear (it should handle its own FP16 math)
        # This returns an FP32 tensor b.c. of HalfLinear implementation
        res_base = super().forward(x) 
        
        # Run LoRA path in FP32
        # Ensuring x is FP32 to match lora_a/b weights...
        x_fp32 = x.to(torch.float32)
        res_lora = self.lora_b(self.lora_a(x_fp32)) * self.scaling
        
        # Sum them (both are now FP32) and return in original x.dtype?
        return (res_base + res_lora).to(x.dtype)

        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        input_dtype = x.dtype
        
        result_base_linear = super().forward(x.to(torch.float16))
        x_imtermediary = x.to(torch.float32)
        x_imtermediary = self.lora_a(x_imtermediary)
        x_imtermediary = self.lora_b(x_imtermediary)
        x_imtermediary = x_imtermediary * self.scaling
        
        return (result_base_linear + x_imtermediary).to(input_dtype)


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            alpha = lora_dim * 5
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim, alpha, False),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim, alpha, False),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim, alpha, False),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
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
        # 1. Freeze EVERYTHING in the network
        for param in self.parameters():
            param.requires_grad = False

        # 2. Unfreeze ONLY the LoRA adapters
        # This looks for any parameter with "lora" in its name
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)



def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    
    return net

if __name__ == "__main__":
    verify_lora_initialization()