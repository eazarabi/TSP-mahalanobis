"""
Encoder-side RDT: loops the ENTIRE encoder stack T times with per-iteration
depth-wise LoRA. Mirrors the OpenMythos/Geiping RDT pattern directly.

Standard encoder:
    input_proj(coords) -> Layer1 -> Layer2 -> Layer3 -> embeddings

Recurrent encoder (this file):
    input_proj(coords) -> h
    for k in 1..T:
        h = Layer1(Layer2(Layer3(h))) + LoRA_k(h)
    -> embeddings

All encoder layer weights are SHARED across iterations (the core RDT property).
Each of the T iterations gets a small rank-8 LoRA adapter to specialize.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from model import TSPEncoder, TSPTransformer, TSPDecoderGlimpse
from experimental.model_rdt import DepthWiseLoRA


class RecurrentEncoder(nn.Module):
    """
    Wraps a standard TSPEncoder and loops its layer stack T times.
    Reuses the same (L=3) layer weights across all T iterations; only the
    per-iteration LoRA adapters differentiate them.
    """
    def __init__(self, base_encoder: TSPEncoder, num_encoder_loops: int = 4,
                 lora_rank: int = 8):
        super().__init__()
        self.base = base_encoder
        self.T = num_encoder_loops
        self.embed_dim = base_encoder.input_proj.out_features
        self.lora_adapters = nn.ModuleList([
            DepthWiseLoRA(self.embed_dim, rank=lora_rank) for _ in range(num_encoder_loops)
        ])

    def forward(self, coords):
        h = self.base.input_proj(coords)
        attn_bias = None
        if self.base.spatial_bias is not None:
            attn_bias = self.base.spatial_bias(coords)

        for k in range(self.T):
            # Apply the full shared encoder stack once per loop iteration
            for layer in self.base.layers:
                h = layer(h, attn_bias=attn_bias)
            # Per-iteration LoRA correction
            h = h + self.lora_adapters[k](h)
        return h


def build_encoder_rdt_model(config, num_encoder_loops: int = 4, lora_rank: int = 8):
    """
    Build a TSPTransformer whose encoder is replaced with a RecurrentEncoder.
    The decoder is the standard TSPDecoderGlimpse (unchanged from main paper).
    """
    model = TSPTransformer(config, use_glimpse=True)
    rec_encoder = RecurrentEncoder(model.encoder, num_encoder_loops, lora_rank)
    model.encoder = rec_encoder
    return model
