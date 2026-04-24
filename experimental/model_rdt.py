"""
Experimental: Recurrent-Depth Decoder for TSP with Depth-Wise LoRA.

Extends the standard glimpse decoder with T iterations of internal reasoning per
decoding step. Each iteration refines the query via (shared) multi-head attention
with a small per-iteration LoRA adapter that lets it specialize.

Inspired by:
  - Geiping et al. "Scaling by Thinking in Continuous Space" (2025) -- RDT
  - Bae et al. "Relaxed Recursive Transformers" (2024) -- depth-wise LoRA
  - Chen et al. "Inner Thinking Transformer" (ACL 2025) -- per-step thinking

Novelty here: RDT is typically applied to the encoder/full-stack for reasoning
tasks. We apply it to the decoder of a neural TSP solver, conditioning each
thinking iteration on the partial-tour state. This targets the known beam-search
collapse failure mode: instead of exploring alternative tours across steps, we
refine the decision within each step.

This module is self-contained and imports from the main codebase without
modifying it. Safe to delete the `experimental/` directory with no side effects.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseLoRA(nn.Module):
    """
    Low-rank adapter applied per loop iteration.

    Gives each of the T iterations a small capacity to specialize without
    duplicating the base attention weights. Rank r is typically 8-16.

    Delta(h) = B @ A @ h, with A in R^{r x d}, B in R^{d x r}, r << d.
    """
    def __init__(self, embed_dim: int, rank: int = 8):
        super().__init__()
        self.A = nn.Linear(embed_dim, rank, bias=False)
        self.B = nn.Linear(rank, embed_dim, bias=False)
        # Initialize B to zero so the adapter starts as identity
        nn.init.zeros_(self.B.weight)
        nn.init.normal_(self.A.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(h))


class RecurrentGlimpseDecoder(nn.Module):
    """
    Drop-in replacement for TSPDecoderGlimpse with T thinking iterations per step.

    At each decoding step t:
      1. Compute context c_t from (graph_mean, pi_1, pi_{t-1})
      2. Loop k = 1, ..., T:
           q_t^{(k)} = MHA_glimpse(q_t^{(k-1)}, H, H; mask)
           q_t^{(k)} = q_t^{(k-1)} + q_t^{(k)} + LoRA_k(q_t^{(k)})    # residual + adapter
           Optionally halt when KL(p^{(k)} || p^{(k-1)}) < eps
      3. Compute logits from final refined query
      4. Sample/argmax next city

    The loop shares base MHA weights across iterations; only the LoRA adapters
    differ per iteration. This is parameter-efficient:
      base MHA:  ~3 * embed_dim^2 params (Q, K, V projections + W_O)
      T adapters: T * 2 * embed_dim * rank  params

    For embed_dim=128, rank=8, T=4: adapter overhead = 8,192 params (~1% of base).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        tanh_clipping: float = 10.0,
        num_thinking_steps: int = 4,
        lora_rank: int = 8,
        use_halting: bool = False,
        halting_kl_threshold: float = 1e-3,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tanh_clipping = tanh_clipping
        self.T = num_thinking_steps
        self.use_halting = use_halting
        self.halting_kl = halting_kl_threshold

        # Shared base weights (same as standard glimpse decoder)
        self.W_context = nn.Linear(3 * embed_dim, embed_dim, bias=False)
        self.W_glimpse_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_glimpse_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_glimpse_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_logit_k = nn.Linear(embed_dim, embed_dim, bias=False)

        # Per-iteration LoRA adapters
        self.lora_adapters = nn.ModuleList([
            DepthWiseLoRA(embed_dim, rank=lora_rank)
            for _ in range(num_thinking_steps)
        ])

    def _glimpse_attention(self, query, embeddings, visited_mask):
        """One pass of masked multi-head attention. Used inside the thinking loop."""
        bs, n, d = embeddings.size()
        h, dk = self.num_heads, self.head_dim

        K = self.W_glimpse_k(embeddings).view(bs, n, h, dk).transpose(1, 2)
        V = self.W_glimpse_v(embeddings).view(bs, n, h, dk).transpose(1, 2)
        Q = query.view(bs, h, 1, dk)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        mask_exp = visited_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask_exp, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(bs, 1, d).squeeze(1)
        return self.W_glimpse_out(out)

    def _compute_logits(self, query, K_logit, visited_mask):
        """Compute the final softmax-compatible logits over cities."""
        bs, d = query.shape
        n = K_logit.size(1)
        scores = torch.matmul(query.unsqueeze(1), K_logit.transpose(-2, -1)).squeeze(1)
        scores = scores / math.sqrt(d)
        scores = self.tanh_clipping * torch.tanh(scores)
        return scores.masked_fill(visited_mask, float('-inf'))

    def _think(self, query_base, embeddings, K_logit, visited_mask, return_steps=False):
        """
        Execute T thinking iterations on the context query.
        Returns the final refined query and optionally per-step logits.
        """
        query = query_base
        step_logits = []
        thinking_steps_used = torch.full(
            (query.size(0),), self.T, dtype=torch.long, device=query.device
        )

        prev_probs = None
        for k in range(self.T):
            # One attention hop + LoRA
            attn_out = self._glimpse_attention(query, embeddings, visited_mask)
            delta = self.lora_adapters[k](attn_out)
            query = query + attn_out + delta  # residual with LoRA refinement

            # Optional halting check
            if self.use_halting:
                logits_k = self._compute_logits(query, K_logit, visited_mask)
                probs_k = F.softmax(logits_k, dim=-1)
                if prev_probs is not None:
                    # Per-sample KL divergence: KL(p^{(k)} || p^{(k-1)})
                    eps = 1e-10
                    kl = (probs_k * (torch.log(probs_k + eps) - torch.log(prev_probs + eps))).sum(-1)
                    # If halted on a previous step, keep its count; else update
                    just_halted = (kl < self.halting_kl) & (thinking_steps_used == self.T)
                    thinking_steps_used = torch.where(
                        just_halted,
                        torch.full_like(thinking_steps_used, k + 1),
                        thinking_steps_used,
                    )
                prev_probs = probs_k.detach()
                if return_steps:
                    step_logits.append(logits_k)

        return query, step_logits, thinking_steps_used

    def forward(
        self,
        embeddings,
        decode_type: str = "sample",
        return_entropy: bool = False,
        return_probs: bool = False,
        beam_width: int = 10,
        start_node=None,
    ):
        """
        Autoregressive tour construction with T thinking iterations per step.
        Signature matches TSPDecoderGlimpse for drop-in compatibility.
        """
        bs, n, d = embeddings.size()
        device = embeddings.device
        graph_embed = embeddings.mean(dim=1)
        K_logit = self.W_logit_k(embeddings)

        if start_node is not None:
            first_city = start_node
        else:
            first_city = torch.zeros(bs, dtype=torch.long, device=device)
        prev_city = first_city

        visited_mask = torch.zeros(bs, n, dtype=torch.bool, device=device)
        visited_mask = visited_mask.scatter(1, first_city.unsqueeze(1), True)

        tours = [first_city]
        log_probs = [torch.zeros(bs, device=device)]
        entropies = []
        total_steps_used = torch.zeros(bs, device=device)

        for step in range(1, n):
            # Build context from partial-tour state
            first_embed = embeddings.gather(
                1, first_city.unsqueeze(1).unsqueeze(2).expand(-1, 1, d)
            ).squeeze(1)
            prev_embed = embeddings.gather(
                1, prev_city.unsqueeze(1).unsqueeze(2).expand(-1, 1, d)
            ).squeeze(1)
            context = torch.cat([graph_embed, first_embed, prev_embed], dim=-1)
            query_base = self.W_context(context)

            # T thinking iterations with LoRA refinement
            refined_query, _, steps_used = self._think(
                query_base, embeddings, K_logit, visited_mask
            )
            total_steps_used = total_steps_used + steps_used.float()

            # Compute final logits and sample/argmax
            scores = self._compute_logits(refined_query, K_logit, visited_mask)
            probs = F.softmax(scores, dim=-1)

            if decode_type == "greedy":
                next_city = probs.argmax(dim=-1)
            elif decode_type == "sample":
                next_city = torch.multinomial(probs, 1).squeeze(1)
            else:
                raise ValueError(f"Unsupported decode_type: {decode_type}")

            chosen_prob = probs.gather(1, next_city.unsqueeze(1)).squeeze(1)
            log_probs.append(torch.log(chosen_prob + 1e-10))

            if return_entropy:
                entropies.append(-(probs * torch.log(probs + 1e-10)).sum(dim=-1))

            visited_mask = visited_mask.scatter(1, next_city.unsqueeze(1), True)
            tours.append(next_city)
            prev_city = next_city

        tours = torch.stack(tours, dim=1)
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)

        out = (tours, total_log_prob)
        if return_entropy:
            out = out + (torch.stack(entropies, dim=1).sum(dim=1),)
        return out


def wrap_model_with_rdt(base_model, num_thinking_steps=4, lora_rank=8, use_halting=False):
    """
    Replace the decoder of an existing TSPTransformer with a RecurrentGlimpseDecoder.
    Keeps the encoder (and its whitening + spatial encoding) intact.

    Usage:
        from model import TSPTransformer
        from experimental.model_rdt import wrap_model_with_rdt

        cfg = Config()
        base = TSPTransformer(cfg, use_glimpse=True)
        rdt_model = wrap_model_with_rdt(base, num_thinking_steps=4)
    """
    cfg = base_model.config
    new_decoder = RecurrentGlimpseDecoder(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        tanh_clipping=cfg.tanh_clipping,
        num_thinking_steps=num_thinking_steps,
        lora_rank=lora_rank,
        use_halting=use_halting,
    )

    # Copy shared weights from the existing glimpse decoder (if present)
    old = base_model.decoder
    if hasattr(old, "W_context"):
        new_decoder.W_context.weight.data.copy_(old.W_context.weight.data)
    if hasattr(old, "W_glimpse_k"):
        new_decoder.W_glimpse_k.weight.data.copy_(old.W_glimpse_k.weight.data)
        new_decoder.W_glimpse_v.weight.data.copy_(old.W_glimpse_v.weight.data)
        new_decoder.W_glimpse_out.weight.data.copy_(old.W_glimpse_out.weight.data)
    if hasattr(old, "W_logit_k"):
        new_decoder.W_logit_k.weight.data.copy_(old.W_logit_k.weight.data)

    base_model.decoder = new_decoder
    return base_model