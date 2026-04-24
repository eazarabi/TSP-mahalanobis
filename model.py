"""
TSP Transformer with:
  - Multi-Head Attention encoder/decoder
  - Spatial Encoding Bias (distance-indexed learnable attention bias)
  - POMO multi-start decoding
  - Glimpse decoder with beam search and entropy support
"""

import math, torch, torch.nn as nn, torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, attn_bias=None):
        batch = query.size(0)
        seq_q = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch, seq_q, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch, -1, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Spatial encoding bias: add distance-based bias to attention logits
        if attn_bias is not None:
            # attn_bias: (batch, num_heads, seq_q, seq_k) or (batch, 1, seq_q, seq_k)
            scores = scores + attn_bias

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(batch, seq_q, self.embed_dim)

        return self.W_o(out)
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim),)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        attn_out = self.self_attn(x, x, x, attn_bias=attn_bias)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class SpatialEncodingBias(nn.Module):
    """
    Learnable distance-indexed bias added to attention logits.
    Discretizes pairwise distances into bins, learns a per-head scalar per bin.
    Inspired by Zhao & Wong (PLOS One, 2025).
    """
    def __init__(self, num_heads, num_bins=32):
        super().__init__()
        self.num_bins = num_bins
        self.num_heads = num_heads
        # Learnable bias: (num_heads, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_heads, num_bins))
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, coords):
        """
        coords: (batch, n, 2)
        Returns: (batch, num_heads, n, n) attention bias
        """
        # Compute pairwise Euclidean distances (works in whitened or raw space)
        diffs = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, n, n, 2)
        dists = diffs.norm(p=2, dim=-1)  # (batch, n, n)

        # Normalize to [0, 1] per-instance and discretize into bins
        max_dist = dists.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        max_dist = max_dist.clamp(min=1e-8)
        normalized = dists / max_dist  # (batch, n, n) in [0, 1]
        bin_idx = (normalized * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

        # Look up bias per head: (num_heads, num_bins) -> (batch, num_heads, n, n)
        bias = self.bias[:, bin_idx]  # (num_heads, batch, n, n)
        bias = bias.permute(1, 0, 2, 3)  # (batch, num_heads, n, n)
        return bias


class TSPEncoder(nn.Module):
    """TSP Encoder maps raw city coordinates to rich embeddings."""
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.0,
                 use_spatial_encoding=False, num_distance_bins=32):
        super().__init__()
        self.input_proj = nn.Linear(2, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layers)])
        self.spatial_bias = None
        if use_spatial_encoding:
            self.spatial_bias = SpatialEncodingBias(num_heads, num_distance_bins)

    def forward(self, coords):
        h = self.input_proj(coords)
        attn_bias = None
        if self.spatial_bias is not None:
            attn_bias = self.spatial_bias(coords)
        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias)
        return h
        
class TSPDecoder(nn.Module):
    """
    Simple pointer decoder.
    forward() supports decode_type in {"sample", "greedy", "beam_search"}.
    Optionally returns entropy and per-step probabilities.
    """
    def __init__(self, embed_dim, num_heads, tanh_clipping=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.W_context = nn.Linear(3*embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, embeddings, decode_type="sample",
                return_entropy=False, return_probs=False, beam_width=10,
                start_node=None):
        if decode_type == "beam_search":
            return self._beam_search(embeddings, beam_width)

        batch_size, n, d = embeddings.size()
        device = embeddings.device
        graph_embed = embeddings.mean(dim=1)
        K = self.W_k(embeddings)

        if start_node is not None:
            first_city = start_node
        else:
            first_city = torch.zeros(batch_size, dtype=torch.long, device=device)
        prev_city = first_city
        visited_mask = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        visited_mask = visited_mask.scatter(1, first_city.unsqueeze(1), True)

        tours = [first_city]
        log_probs = [torch.zeros(batch_size, device=device)]
        entropies = []
        all_probs = []

        for step in range(1, n):
            first_embed = embeddings.gather(1, first_city.unsqueeze(1).unsqueeze(2).expand(-1,1,d)).squeeze(1)
            prev_embed = embeddings.gather(1, prev_city.unsqueeze(1).unsqueeze(2).expand(-1,1,d)).squeeze(1)
            context = torch.cat([graph_embed, first_embed, prev_embed], dim=-1)
            query = self.W_context(context)

            scores = torch.matmul(query.unsqueeze(1), K.transpose(-2,-1)).squeeze(1)
            scores = scores / math.sqrt(d)
            scores = self.tanh_clipping * torch.tanh(scores)
            scores = scores.masked_fill(visited_mask, float('-inf'))
            probs = F.softmax(scores, dim=-1)

            if decode_type == "greedy":
                next_city = probs.argmax(dim=-1)
            elif decode_type == "sample":
                next_city = torch.multinomial(probs, 1).squeeze(1)
            else:
                raise ValueError(f"Invalid decode_type: {decode_type}")

            chosen_prob = probs.gather(1, next_city.unsqueeze(1)).squeeze(1)
            log_probs.append(torch.log(chosen_prob + 1e-10))

            if return_entropy:
                step_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropies.append(step_entropy)
            if return_probs:
                all_probs.append(probs)

            visited_mask = visited_mask.scatter(1, next_city.unsqueeze(1), True)
            tours.append(next_city)
            prev_city = next_city

        tours = torch.stack(tours, dim=1)
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)

        out = (tours, total_log_prob)
        if return_entropy:
            total_entropy = torch.stack(entropies, dim=1).sum(dim=1)
            out = out + (total_entropy,)
        if return_probs:
            all_probs_t = torch.stack(all_probs, dim=1)
            out = out + (all_probs_t,)
        return out
    
    def _beam_search(self, embeddings, beam_width):
        """Beam search decoding. Returns best tour per instance + log-prob."""
        batch_size, n, d = embeddings.size()
        device = embeddings.device
        B = beam_width
        
        graph_embed = embeddings.mean(dim=1)                    # (batch, d)
        K = self.W_k(embeddings)                                # (batch, n, d)
        
        # Initialize: all beams start at city 0
        tours = torch.zeros(batch_size, B, 1, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, B, device=device)
        visited = torch.zeros(batch_size, B, n, dtype=torch.bool, device=device)
        visited.scatter_(2, tours, True)
        
        for step in range(1, n):
            prev_city = tours[:, :, -1]                         # (batch, B)
            first_city_idx = tours[:, :, 0]                     # (batch, B)
            
            # Expand embeddings per beam
            emb_exp = embeddings.unsqueeze(1).expand(-1, B, -1, -1)  # (batch, B, n, d)
            prev_embed = emb_exp.gather(
                2, prev_city.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,d)
            ).squeeze(2)                                        # (batch, B, d)
            first_embed = emb_exp.gather(
                2, first_city_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,d)
            ).squeeze(2)                                        # (batch, B, d)
            graph_exp = graph_embed.unsqueeze(1).expand(-1, B, -1)
            
            context = torch.cat([graph_exp, first_embed, prev_embed], dim=-1)  # (batch, B, 3d)
            query = self.W_context(context)                     # (batch, B, d)
            
            K_exp = K.unsqueeze(1).expand(-1, B, -1, -1)        # (batch, B, n, d)
            scores = torch.matmul(query.unsqueeze(2), K_exp.transpose(-2,-1)).squeeze(2)
            scores = scores / math.sqrt(d)
            scores = self.tanh_clipping * torch.tanh(scores)
            scores = scores.masked_fill(visited, float('-inf'))
            
            log_probs_step = F.log_softmax(scores, dim=-1)      # (batch, B, n)
            
            # Candidate scores: add step log-prob to running log-prob
            candidate = log_probs.unsqueeze(-1) + log_probs_step  # (batch, B, n)
            candidate_flat = candidate.view(batch_size, -1)      # (batch, B*n)
            
            top_log_probs, top_idx = candidate_flat.topk(B, dim=-1)
            beam_idx = top_idx // n                              # (batch, B)
            city_idx = top_idx % n                               # (batch, B)
            
            # Reassemble tours and visited mask per new beam
            beam_idx_exp = beam_idx.unsqueeze(-1).expand(-1, -1, step)
            selected_tours = tours.gather(1, beam_idx_exp)
            tours = torch.cat([selected_tours, city_idx.unsqueeze(-1)], dim=-1)
            
            visited = visited.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, n))
            visited = visited.scatter(2, city_idx.unsqueeze(-1), True)
            
            log_probs = top_log_probs
        
        # Best beam per instance
        best_beam = log_probs.argmax(dim=-1)                     # (batch,)
        batch_range = torch.arange(batch_size, device=device)
        final_tours = tours[batch_range, best_beam]              # (batch, n)
        final_log_probs = log_probs[batch_range, best_beam]      # (batch,)
        return final_tours, final_log_probs


class TSPDecoderGlimpse(nn.Module):
    """Glimpse-enhanced decoder with beam search and entropy support."""
    def __init__(self, embed_dim, num_heads, tanh_clipping=10.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tanh_clipping = tanh_clipping
        self.W_context = nn.Linear(3 * embed_dim, embed_dim, bias=False)
        self.W_glimpse_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_glimpse_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_glimpse_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_logit_k = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def _compute_query(self, embeddings, first_city, prev_city, graph_embed,
                       visited_mask):
        """Shared: context → refined query via glimpse attention."""
        batch_size, n, d = embeddings.size()
        h = self.num_heads
        dk = self.head_dim
        
        first_embed = embeddings.gather(1, first_city.unsqueeze(1).unsqueeze(2).expand(-1, 1, d)).squeeze(1)
        prev_embed = embeddings.gather(1, prev_city.unsqueeze(1).unsqueeze(2).expand(-1, 1, d)).squeeze(1)
        context = torch.cat([graph_embed, first_embed, prev_embed], dim=-1)
        query_base = self.W_context(context)
        
        K_glimpse = self.W_glimpse_k(embeddings).view(batch_size, n, h, dk).transpose(1, 2)
        V_glimpse = self.W_glimpse_v(embeddings).view(batch_size, n, h, dk).transpose(1, 2)
        
        Q_glimpse = query_base.view(batch_size, h, 1, dk)
        gs = torch.matmul(Q_glimpse, K_glimpse.transpose(-2, -1)) / math.sqrt(dk)
        mask_exp = visited_mask.unsqueeze(1).unsqueeze(2)
        gs = gs.masked_fill(mask_exp, float('-inf'))
        ga = F.softmax(gs, dim=-1)
        gh = torch.matmul(ga, V_glimpse)
        gc = gh.transpose(1, 2).contiguous().view(batch_size, 1, d).squeeze(1)
        return self.W_glimpse_out(gc)
    
    def forward(self, embeddings, decode_type="sample",
                return_entropy=False, return_probs=False, beam_width=10,
                start_node=None):
        if decode_type == "beam_search":
            return self._beam_search(embeddings, beam_width)

        batch_size, n, d = embeddings.size()
        device = embeddings.device
        graph_embed = embeddings.mean(dim=1)
        K_logit = self.W_logit_k(embeddings)

        # POMO: allow specifying start node per instance
        if start_node is not None:
            first_city = start_node
        else:
            first_city = torch.zeros(batch_size, dtype=torch.long, device=device)
        prev_city = first_city
        visited_mask = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        visited_mask = visited_mask.scatter(1, first_city.unsqueeze(1), True)

        tours = [first_city]
        log_probs = [torch.zeros(batch_size, device=device)]
        entropies = []
        all_probs = []

        for step in range(1, n):
            query_refined = self._compute_query(embeddings, first_city, prev_city,
                                                 graph_embed, visited_mask)

            scores = torch.matmul(query_refined.unsqueeze(1), K_logit.transpose(-2,-1)).squeeze(1)
            scores = scores / math.sqrt(d)
            scores = self.tanh_clipping * torch.tanh(scores)
            scores = scores.masked_fill(visited_mask, float('-inf'))
            probs = F.softmax(scores, dim=-1)

            if decode_type == "greedy":
                next_city = probs.argmax(dim=-1)
            elif decode_type == "sample":
                next_city = torch.multinomial(probs, 1).squeeze(1)
            else:
                raise ValueError(f"Invalid decode_type: {decode_type}")

            chosen_prob = probs.gather(1, next_city.unsqueeze(1)).squeeze(1)
            log_probs.append(torch.log(chosen_prob + 1e-10))

            if return_entropy:
                step_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropies.append(step_entropy)
            if return_probs:
                all_probs.append(probs)

            visited_mask = visited_mask.scatter(1, next_city.unsqueeze(1), True)
            tours.append(next_city)
            prev_city = next_city

        tours = torch.stack(tours, dim=1)
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)

        out = (tours, total_log_prob)
        if return_entropy:
            total_entropy = torch.stack(entropies, dim=1).sum(dim=1)
            out = out + (total_entropy,)
        if return_probs:
            all_probs_t = torch.stack(all_probs, dim=1)
            out = out + (all_probs_t,)
        return out
    
    def _beam_search(self, embeddings, beam_width):
        batch_size, n, d = embeddings.size()
        device = embeddings.device
        B = beam_width
        
        graph_embed = embeddings.mean(dim=1)
        K_logit = self.W_logit_k(embeddings)
        
        tours = torch.zeros(batch_size, B, 1, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, B, device=device)
        visited = torch.zeros(batch_size, B, n, dtype=torch.bool, device=device)
        visited.scatter_(2, tours, True)
        
        # Expand embeddings to per-beam via flattening batch×beam dim
        for step in range(1, n):
            # Flatten (batch, B, ...) → (batch*B, ...) for decoder reuse
            bb = batch_size * B
            emb_bb = embeddings.unsqueeze(1).expand(-1, B, -1, -1).reshape(bb, n, d)
            graph_bb = graph_embed.unsqueeze(1).expand(-1, B, -1).reshape(bb, d)
            prev_bb = tours[:, :, -1].reshape(bb)
            first_bb = tours[:, :, 0].reshape(bb)
            visited_bb = visited.reshape(bb, n)
            
            query_refined = self._compute_query(emb_bb, first_bb, prev_bb,
                                                 graph_bb, visited_bb)
            K_bb = K_logit.unsqueeze(1).expand(-1, B, -1, -1).reshape(bb, n, d)
            
            scores = torch.matmul(query_refined.unsqueeze(1), K_bb.transpose(-2,-1)).squeeze(1)
            scores = scores / math.sqrt(d)
            scores = self.tanh_clipping * torch.tanh(scores)
            scores = scores.masked_fill(visited_bb, float('-inf'))
            
            log_probs_step = F.log_softmax(scores, dim=-1).view(batch_size, B, n)
            
            candidate = log_probs.unsqueeze(-1) + log_probs_step
            candidate_flat = candidate.view(batch_size, -1)
            
            top_log_probs, top_idx = candidate_flat.topk(B, dim=-1)
            beam_idx = top_idx // n
            city_idx = top_idx % n
            
            beam_idx_exp = beam_idx.unsqueeze(-1).expand(-1, -1, step)
            selected_tours = tours.gather(1, beam_idx_exp)
            tours = torch.cat([selected_tours, city_idx.unsqueeze(-1)], dim=-1)
            
            visited = visited.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, n))
            visited = visited.scatter(2, city_idx.unsqueeze(-1), True)
            
            log_probs = top_log_probs
        
        best_beam = log_probs.argmax(dim=-1)
        batch_range = torch.arange(batch_size, device=device)
        final_tours = tours[batch_range, best_beam]
        final_log_probs = log_probs[batch_range, best_beam]
        return final_tours, final_log_probs


class TSPTransformer(nn.Module):
    """Pass-through wrapper to encoder+decoder with all kwargs forwarded."""
    def __init__(self, config, use_glimpse=False):
        super().__init__()
        self.config = config
        self.use_glimpse = use_glimpse
        use_spatial = getattr(config, 'use_spatial_encoding', False)
        num_bins = getattr(config, 'num_distance_bins', 32)
        self.encoder = TSPEncoder(
            embed_dim=config.embed_dim, num_heads=config.num_heads,
            ff_dim=config.ff_dim, num_layers=config.num_encoder_layers,
            dropout=config.dropout,
            use_spatial_encoding=use_spatial,
            num_distance_bins=num_bins,
        )
        if use_glimpse:
            self.decoder = TSPDecoderGlimpse(
                embed_dim=config.embed_dim, num_heads=config.num_heads,
                tanh_clipping=config.tanh_clipping,
            )
        else:
            self.decoder = TSPDecoder(
                embed_dim=config.embed_dim, num_heads=config.num_heads,
                tanh_clipping=config.tanh_clipping,
            )

    def forward(self, coords, decode_type="sample",
                return_entropy=False, return_probs=False, beam_width=10,
                start_node=None):
        embeddings = self.encoder(coords)
        return self.decoder(embeddings, decode_type=decode_type,
                             return_entropy=return_entropy,
                             return_probs=return_probs,
                             beam_width=beam_width,
                             start_node=start_node)

class CriticNetwork(nn.Module):
    """
    Critic Network Vφ(s) for the critic baseline.
    Predicts expected tour length given city coordinates.
    
    Trained via MSE: L(φ) = (1/B) Σ ||L(πⁱ) - Vφ(sⁱ)||²
    Used in the ablation study comparing rollout vs critic baselines.
    """

    def __init__(self, config):
        super().__init__()
        d = config.critic_embed_dim

        # Reuse the encoder architecture — but these are separate weights
        self.encoder = TSPEncoder(
            embed_dim=d,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.critic_num_layers,
            dropout=config.dropout,
        )

        # Head: maps pooled embedding to predicted tour length (scalar)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, coords):
        """
        Predict expected tour length for a batch of TSP instances.
        """
        embeddings = self.encoder(coords)    # (batch, n, d)
        pooled = embeddings.mean(dim=1)       # (batch, d) — average across cities
        return self.head(pooled).squeeze(-1)  # (batch,) — squeeze out singleton dim