# Transformer Architecture for MADMESHR: Design Notes

**Date:** 2026-04-07 (Session 16)
**Status:** Design sketch — no code

## Motivation

The current DQN uses a fixed 44-float state vector encoding local boundary context (neighbor positions, angles, fan samples, area ratio). This works for domains up to ~12 vertices but struggles on larger boundaries (29v+) because:

1. **Fixed context window:** Only sees immediate neighbors of the reference vertex, missing global boundary structure
2. **No attention over distant vertices:** Can't reason about type-2 proximity pairs that are geometrically close but topologically distant
3. **Single reference vertex:** Must commit to one vertex before seeing the action space — bad for large boundaries where the best action may be far from the current reference

## Proposed Architecture

### Input: Variable-Length Boundary Sequence

Replace the 44-float vector with a sequence of per-vertex feature vectors:

```
For each boundary vertex v_i (i = 0..N-1):
  - Position (x, y): 2 floats (normalized to domain bounding box)
  - Interior angle at v_i: 1 float (normalized to [-1, 1] where 0 = pi)
  - Edge lengths (to prev, to next): 2 floats (normalized by mean edge length)
  - Distance to nearest non-adjacent vertex: 1 float (type-2 signal)
  - Is reference vertex: 1 float (binary)
  - Local curvature (angle change rate): 1 float
  - Distance to original boundary: 1 float (concavity signal)
```

**Per-vertex features: 9 floats.** Sequence length = N (number of boundary vertices).

### Global Context Token

Prepend a [CLS]-style token with global features:
- Total boundary vertices (N, normalized)
- Total area remaining (normalized by initial area)
- Number of elements placed so far
- Number of pending loops
- Completion fraction

**Global features: 5 floats.**

### Encoder: Transformer with Rotary Position Embeddings

```
Input: [CLS, v_0, v_1, ..., v_{N-1}]  shape: (N+1, d_model)

Embedding: Linear(9 → d_model) for vertices, Linear(5 → d_model) for CLS
Position: Rotary embeddings (RoPE) on the cyclic boundary — vertex i gets position i/N * 2π

Transformer blocks (L=4, d_model=64, n_heads=4, d_ff=128):
  - Multi-head self-attention (vertices attend to all other vertices)
  - Feed-forward: Linear → ReLU → Linear
  - LayerNorm + residual connections
  - Dropout 0.1
```

**Key design choice: Cyclic RoPE.** The boundary is a closed polygon, so position embeddings should be cyclic. Standard RoPE encodes position as angles — for a cyclic boundary, vertex positions map naturally to angles on a circle. This means vertex 0 and vertex N-1 are neighbors in the embedding space, as they should be.

### Decoder: Dual-Head (Value + Advantage)

**Dueling architecture preserved** from current DQN:

```
CLS output → Value head: Linear(d_model, d_model/2) → ReLU → Linear(d_model/2, 1)

Per-vertex outputs → Advantage head:
  For each vertex v_i:
    - Type-0 advantage: Linear(d_model, 1)  [connect v_i to v_{i+2}]
    - Type-1 advantages: Linear(d_model, n_angle * n_dist) [place interior vertex]
    - Type-2 advantages: Linear(2 * d_model, 1) per proximity pair [concat v_i and v_j features]

Q(s, a) = V(s) + A(s, a) - mean_valid(A)
```

**Critical change: Multi-vertex action space.** Every vertex is a potential reference vertex, not just the pre-selected one. The transformer outputs per-vertex features that directly parameterize advantages for actions rooted at that vertex.

This naturally implements **multi-vertex selection** — the agent chooses both WHERE to act (which vertex) and WHAT to do (type-0/1/2 action) in a single forward pass.

### Action Space Size

For N boundary vertices with 12-angle × 4-dist type-1 grid:
- Type-0: N actions (one per vertex)
- Type-1: N × 48 actions
- Type-2: variable (one per proximity pair)
- Total: ~49N + n_pairs

For N=29: ~1421 + pairs ≈ 1500 actions. Manageable with masking.

For N=64: ~3136 + pairs ≈ 3200 actions. Still manageable.

### Training Considerations

1. **Curriculum:** Start with small domains (square, octagon), transfer to larger ones. The transformer architecture should transfer better than the fixed-vector DQN because the per-vertex features are domain-size-invariant.

2. **Batch padding:** Variable-length sequences need padding + attention masks. Use a maximum sequence length (128 vertices) and pad shorter boundaries.

3. **Computational cost:** Self-attention is O(N²) per layer. For N ≤ 128, this is negligible compared to the geometry computations in the environment. Estimated: ~2x slower than current DQN forward pass for N=64.

4. **Replay buffer:** Store (boundary_vertices, action, reward, next_boundary_vertices, action_mask, done). Variable-length — use list-based buffer, not fixed-size arrays.

## Comparison to QuadGPT (2026)

QuadGPT uses an autoregressive transformer to generate quad meshes token-by-token (vertex coordinates as tokens). Key differences from our approach:

| Aspect | QuadGPT | MADMESHR Transformer |
|--------|---------|---------------------|
| Paradigm | Autoregressive generation | Advancing-front RL |
| Input | Point cloud / boundary | Current boundary state |
| Output | Vertex coordinate tokens | Discrete action (vertex + placement) |
| Training | Supervised + tDPO | DQN with action masking |
| Topology guarantee | None (post-processing) | By construction (advancing front) |
| Variable mesh size | Yes (autoregressive) | Yes (variable boundary length) |

Our approach guarantees valid topology by construction (advancing front always produces valid elements), while QuadGPT needs post-processing to fix topological errors. The transformer here replaces only the state encoder and action selector, not the mesh generation mechanism.

## Migration Path

1. **Phase 1 (session 17-18):** Implement transformer encoder + multi-vertex action space. Train on square and octagon. Verify it matches or beats current DQN on small domains.

2. **Phase 2 (session 19-20):** Train on star, L-shape, H-shape. Evaluate whether transformer quality exceeds the geometry ceiling of the fixed-vector approach.

3. **Phase 3 (session 21+):** Train on full annulus-layer2 directly (no sub-loop decomposition needed if multi-vertex selection works). Compare to sub-loop assembly approach.

## Open Questions

- **Positional encoding:** Is cyclic RoPE the right choice, or should we use learned positional embeddings? The boundary is cyclic but not periodic in the Fourier sense.
- **Attention patterns:** Should we restrict attention to local neighborhoods (sparse attention) for efficiency, or is full self-attention necessary for type-2 reasoning?
- **Action masking at scale:** With ~3000 actions for 64v, computing the full mask every step is expensive. Can we mask lazily (only compute validity for top-K Q-value actions)?
- **Transfer learning:** Can a transformer trained on multiple small domains generalize to unseen larger domains without fine-tuning?
