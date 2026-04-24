"""
Combined RDT: both the encoder stack AND the decoder glimpse attention loop.

This is the most faithful TSP adaptation of the full OpenMythos/RDT paradigm.
The encoder iteratively refines node embeddings over T_enc loops, and at each
decoding step the decoder iteratively refines its query over T_dec loops.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import TSPTransformer
from experimental.model_rdt import RecurrentGlimpseDecoder
from experimental.encoder_rdt.model import RecurrentEncoder


def build_combined_rdt_model(config, num_encoder_loops: int = 4,
                              num_decoder_loops: int = 4, lora_rank: int = 8):
    """
    Build a TSPTransformer with BOTH encoder and decoder looped.
    """
    model = TSPTransformer(config, use_glimpse=True)
    # Replace encoder
    model.encoder = RecurrentEncoder(model.encoder, num_encoder_loops, lora_rank)
    # Replace decoder
    model.decoder = RecurrentGlimpseDecoder(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        tanh_clipping=config.tanh_clipping,
        num_thinking_steps=num_decoder_loops,
        lora_rank=lora_rank,
    )
    return model
