from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    trying 3 bit quantization instead of 4 bit to save more memory
    basically same idea as 4bit but only use 0-7 instead of 0-15
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    # Get the max value like in the 4bit version
    normalization = x.abs().max(dim=-1, keepdim=True).values
    # normalize values between 0 and 1
    x_norm = (x + normalization) / (2 * normalization)
    # Now quantize to 0-7 (3 bits) instead of 0-15
    x_quant = (x_norm * 7).round().to(torch.uint8)
    
    # Need to pack the 3-bit values into bytes
    # i'm packing 8 values (24 bits total) into 3 bytes
    packed_size = (group_size * 3) // 8
    packed = torch.zeros(x_quant.size(0), packed_size, dtype=torch.uint8, device=x.device)
    
    # pack each value into the right position
    for i in range(group_size):
        byte_idx = (i * 3) // 8
        bit_offset = (i * 3) % 8
        
        if bit_offset <= 5:
            # Value fits in current byte
            packed[:, byte_idx] |= (x_quant[:, i] << bit_offset)
        else:
            # value spans two bytes, need to split it
            packed[:, byte_idx] |= (x_quant[:, i] << bit_offset) & 0xFF
            packed[:, byte_idx + 1] |= (x_quant[:, i] >> (8 - bit_offset))
    
    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    reverse the 3bit quantization
    """
    assert packed.dim() == 2
    
    normalization = normalization.to(torch.float32)
    x_quant = torch.zeros(packed.size(0), group_size, dtype=torch.uint8, device=packed.device)
    
    # Unpack each 3-bit value
    for i in range(group_size):
        byte_idx = (i * 3) // 8
        bit_offset = (i * 3) % 8
        
        if bit_offset <= 5:
            x_quant[:, i] = (packed[:, byte_idx] >> bit_offset) & 0x07
        else:
            # Reconstruct from two bytes
            low_bits = (packed[:, byte_idx] >> bit_offset) & ((1 << (8 - bit_offset)) - 1)
            high_bits = (packed[:, byte_idx + 1] & ((1 << (bit_offset - 5)) - 1)) << (8 - bit_offset)
            x_quant[:, i] = low_bits | high_bits
    
    # convert back to normalized values then denormalize
    x_norm = x_quant.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # Figure out how much space we need for packed weights
        total_elements = out_features * in_features
        num_groups = total_elements // group_size
        packed_size = (group_size * 3) // 8

        # Register buffers like in the 4bit version
        self.register_buffer(
            "weight_q3",
            torch.zeros(num_groups, packed_size, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(num_groups, 1, dtype=torch.float16),
            persistent=False,
        )
        
        # hook for loading weights from checkpoint
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        
        # Also use float16 for bias to save more space
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight_key = f"{prefix}weight"
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            # Quantize the weights when loading
            w_q3, w_norm = block_quantize_3bit(weight.flatten(), group_size=self._group_size)
            self.weight_q3.copy_(w_q3)
            self.weight_norm.copy_(w_norm)
            del state_dict[f"{prefix}weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # dequantize weights
            w_flat = block_dequantize_3bit(self.weight_q3, self.weight_norm, self._group_size)
            w = w_flat.view(self._shape)
            
            # Do computation in float16 for speed
            x_half = x.to(torch.float16)
            bias_half = self.bias if self.bias is None else self.bias
            result = torch.nn.functional.linear(x_half, w.to(torch.float16), bias_half)
            return result.to(x.dtype)


class BigNetLowerPrecision(torch.nn.Module):
    """
    BigNet with 3-bit weights to get under 9MB
    using bigger groups (32 instead of 16) to reduce overhead from normalization
    """

    class Block(torch.nn.Module):
        def __init__(self, channels, group_size=32):
            super().__init__()
            # Same structure as regular bignet but with 3bit linear layers
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size=32),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size=32),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size=32),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size=32),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size=32),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size=32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    net = BigNetLowerPrecision()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net