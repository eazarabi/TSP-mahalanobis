# Research Plan: Recurrent-Depth Decoder for Neural TSP

**Status:** Experimental / Future Work
**Files:** `experimental/model_rdt.py`, `experimental/train_rdt.py`, `experimental/evaluate_rdt.py`
**Removable:** Yes — delete the `experimental/` directory to revert fully.

---

## 1. Core Hypothesis

Standard neural TSP solvers (Attention Model, POMO, our whitening approach) use a decoder that makes each city-selection decision in **one forward pass of attention**. This is a plausible point of weakness:

- Each decoding step is a combinatorial choice over $n-t$ candidates.
- The choice must balance local proximity, tour-closure effects, and future consequences.
- A single MHA + tanh does not constitute deep reasoning — it is one inner-product-based scoring pass.
- Our ablation (Section 7.5) showed **beam search collapses** because the policy becomes too peaked. Adding diversity *across steps* does not help; the problem is that each step's decision is shallow.

**Hypothesis:** Looping the decoder $T$ times at each step (with small per-iteration LoRA adapters) produces more careful selections without collapsing the policy. This is "thinking in latent space" applied to TSP construction.

## 2. Why This Is Novel

RDT-style architectures (OpenMythos, Geiping et al. 2025) are designed for language/reasoning tasks and loop the **encoder** or **full stack**. Applying the same idea to the TSP decoder is novel because:

1. **The encoder's job is static:** node embeddings don't change across decoding steps. Looping the encoder is wasted compute.
2. **The decoder's state is dynamic:** at each step, the partial tour grows. Looping the decoder gives the model multiple passes to weigh the growing commitment against remaining options.
3. **Combinatorial decisions benefit from iteration:** unlike autoregressive token generation where each choice is typically local, TSP tour closure means early decisions constrain late ones. Multi-pass reasoning is natural here.

To our knowledge, no prior neural combinatorial optimization work applies RDT to the decoder.

## 3. Mathematical Formulation

### Standard Glimpse Decoder (baseline)

For step $t$:
$$
c_t = W_c [\bar{h} \;\|\; h_{\pi_1} \;\|\; h_{\pi_{t-1}}]
$$
$$
q_t = \mathrm{MHA}(c_t, H, H; \mathrm{mask})
$$
$$
u_j^{(t)} = C \cdot \tanh(q_t^\top k_j' / \sqrt{d}) \text{ if } j \text{ unvisited, else } -\infty
$$
$$
p_\theta(\pi_t = j \mid s_t) = \mathrm{softmax}(u^{(t)})_j
$$

### Proposed: Recurrent Glimpse Decoder

Introduce $T$ thinking iterations per step, with per-iteration LoRA adapters $\Delta_k$:

$$
q_t^{(0)} = c_t
$$
$$
\text{For } k = 1, \ldots, T:\quad g_k = \mathrm{MHA}(q_t^{(k-1)}, H, H; \mathrm{mask})
$$
$$
q_t^{(k)} = q_t^{(k-1)} + g_k + \Delta_k(g_k)
$$
$$
u_j^{(t)} = C \cdot \tanh(q_t^{(T)\top} k_j' / \sqrt{d})
$$

Where $\Delta_k(h) = B_k A_k h$ is a low-rank adapter with $A_k \in \mathbb{R}^{r \times d}$, $B_k \in \mathbb{R}^{d \times r}$, $r \ll d$. Base MHA weights are **shared** across iterations.

### Optional: KL-Divergence Halting

Halt the thinking loop early when successive iterations produce similar distributions:
$$
\mathrm{halt}(k) \iff D_{KL}\bigl(p_t^{(k)} \| p_t^{(k-1)}\bigr) < \epsilon
$$

This gives per-step adaptive compute: easy decisions halt fast, hard ones use the full budget.

### Parameter Count

For $d = 128$, $r = 8$, $T = 4$:
- Base MHA + context: $\sim 3 \cdot 128^2 + 3d \cdot d \approx 98{,}000$ params
- LoRA adapters: $T \cdot 2 \cdot d \cdot r = 4 \cdot 2 \cdot 128 \cdot 8 = 8{,}192$ params
- **Overhead: $\sim 1\%$ of base decoder parameters.**

## 4. Experimental Protocol

### Phase 1: Sanity Checks (Euclidean only, quick)

```bash
# Baseline: T=1 (equivalent to standard glimpse decoder)
python experimental/train_rdt.py --metric euclidean --thinking-steps 1 --epochs 30

# T=2, T=4 with LoRA
python experimental/train_rdt.py --metric euclidean --thinking-steps 2 --epochs 30
python experimental/train_rdt.py --metric euclidean --thinking-steps 4 --epochs 30

# Evaluate
python experimental/evaluate_rdt.py --metric euclidean --thinking-steps 1 2 4
```

**Expected outcomes:**
- Greedy gap for $T=4$ should match or slightly improve $T=1$ (modest gain at best).
- POMO-20 gap should improve more significantly because thinking iterations compound with diverse starts.
- Inference time scales roughly linearly with $T$.

### Phase 2: Mahalanobis + Whitening Synergy

The real payoff hypothesis: whitening produces isotropic geometry, RDT iteratively refines spatial reasoning in that geometry.

```bash
python experimental/train_rdt.py --metric mahalanobis --thinking-steps 4 --epochs 50
python experimental/evaluate_rdt.py --metric mahalanobis --thinking-steps 1 2 4
```

**Success criterion:** RDT + Whitening + POMO-20 + 2-opt achieves a Mahalanobis gap of $\leq 0.25\%$ (our current best: 0.31%).

### Phase 3: Warm-Start from Pre-Trained Model

RDT's LoRA adapters can be added to a trained model and fine-tuned briefly:

```bash
python experimental/train_rdt.py \
    --metric mahalanobis --thinking-steps 4 --epochs 10 \
    --init-from checkpoints/glimpse_rollout_mahalanobis_whiten_best.pt
```

**Hypothesis:** Warm-starting + short RDT fine-tuning outperforms training from scratch because the base weights already encode good representations; LoRA adapters only need to learn the refinement pattern.

### Phase 4: Halting Ablation

```bash
python experimental/train_rdt.py --metric euclidean --thinking-steps 8 --use-halting
```

Measure the average number of thinking steps used per decoding step. If the model genuinely halts early on easy steps, this provides computational savings at inference.

## 5. Risks and Failure Modes

| Risk | Mitigation |
|------|-----------|
| LoRA adapters collapse to zero (identity) | Check training loss curves; consider orthogonality regularization |
| Training instability from residual accumulation over $T$ loops | Layer norm after each residual, gradient clipping |
| $T=4$ just equivalent to $T=1$ (no benefit) | Try $T=8$; verify with per-iteration probes |
| Inference time grows $T\times$ with no quality gain | Activate halting; target adaptive compute budget |
| Doesn't combine well with POMO multi-start | Test RDT-only baseline before integrating |

## 6. Extensions Worth Exploring

1. **Cross-step LoRA memory:** share LoRA state across $t$ (not just $k$) so the decoder accumulates tour-specific "thinking patterns."
2. **Per-token thinking budget (ITT-style):** instead of fixed $T$, route each decoding step through a small MLP that predicts how many iterations it needs.
3. **MoE in the decoder:** small mixture of experts in the feed-forward portion of the glimpse, where each loop iteration routes through different experts.
4. **Joint encoder-decoder looping:** share a single "thinking budget" across encoder refinement and decoder reasoning.

## 7. Theoretical Justification

The "latent BFS" argument (from the OpenMythos video/paper):

> A real-valued vector can represent multiple possible next reasoning directions simultaneously. A discrete token cannot — it collapses to one choice.

For TSP, this maps cleanly:
- At step $t$, there are $n - t$ possible next cities.
- A single attention pass commits to one via softmax.
- $T$ attention passes in latent space can refine the distribution over all of them, weighing tradeoffs that are hard to express in a single pass.

This is what beam search was supposed to do but failed at in our experiments — the policy was already too committed. RDT refines **before** committing.

## 8. Continuation Prompt (for future sessions)

If this direction is picked up later, here is a self-contained prompt:

---

> I am extending a Transformer-based RL TSP solver with a Recurrent-Depth (RDT) decoder as future work beyond the main paper on whitening-transform preprocessing for Mahalanobis TSP. The core code is in `experimental/model_rdt.py`. The idea is to loop the decoder's glimpse attention $T$ times per decoding step, with per-iteration depth-wise LoRA adapters, optionally with KL-divergence halting for adaptive compute. The main paper achieves 0.31% gap on Mahalanobis TSP-20 (POMO + 2-opt); the target for RDT is $\leq 0.25\%$.
>
> Tasks:
> 1. Verify the existing `RecurrentGlimpseDecoder` implementation (mathematical correctness, gradient flow through the loop, LoRA parameter efficiency).
> 2. Run the experimental protocol in `RESEARCH_PLAN.md` sections 4.1–4.4. Report per-$T$ gap, inference time scaling, and halting statistics (if enabled).
> 3. If baseline RDT doesn't improve over the glimpse decoder, diagnose: check LoRA adapter weight norms, attention entropy across iterations, per-step KL divergence trajectories. Propose an architectural fix.
> 4. If RDT improves, investigate extensions: MoE in the decoder feed-forward, per-token adaptive $T$ (ITT-style), warm-starting from the pre-trained whitened Mahalanobis model.
> 5. Write results as either (a) an appendix addition to the existing paper, or (b) a separate follow-up paper on RDT for neural CO.
>
> The main codebase (in the project root) is frozen and should not be modified — `experimental/` is a self-contained extension.

---

## 9. Citations (to add to `custom.bib` if this becomes a paper section)

```bibtex
@article{geiping2025scaling,
    title={Scaling by Thinking in Continuous Space: Recurrent-Depth Transformers},
    author={Geiping, Jonas and others},
    journal={arXiv preprint arXiv:2502.05171},
    year={2025}
}

@article{bae2024relaxed,
    title={Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA},
    author={Bae, Sangmin and others},
    journal={arXiv preprint},
    year={2024}
}

@inproceedings{chen2025inner,
    title={Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking},
    author={Chen, Yilong and others},
    booktitle={ACL},
    year={2025}
}
```