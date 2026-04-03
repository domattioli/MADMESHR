# MADMESHR Training Assessment

## Training Curves

### Phase 3: 50k DQN on Star Domain (10 vertices)
| Timestep | Eval Return | Completion |
|----------|-------------|------------|
| 5,000    | 21.34       | 100%       |
| 10,000   | 19.45       | 100%       |
| 15,000   | 20.51       | 100%       |
| 20,000   | 25.14       | 100%       |
| 25,000   | 21.35       | 100%       |
| 30,000   | 29.38       | 100%       |
| 35,000   | 34.17       | 100%       |
| 40,000   | 46.02       | 100%       |
| 45,000   | 57.56       | 100%       |
| 50,000   | 35.82       | 100%       |

**Best eval return: 57.56** (at 45k). Returns trend upward with high variance. 100% completion throughout.

### Phase 4: 100k DQN on Star Domain (killed at ~38k due to 45-min CPU constraint)
| Timestep | Eval Return | Completion |
|----------|-------------|------------|
| 10,000   | 13.21       | 100%       |
| 20,000   | 37.91       | 100%       |
| 30,000   | 23.66       | 0%         |

Training still in exploration phase (epsilon=0.57 at 30k). The 50k Phase 3 model remains best.

## Mesh Quality

Final eval on star domain (best model):
- **43 quad elements**, 0 triangles
- **Mean equiangular quality: 0.349**
- Quality is moderate; ideal quads would score ~0.7+
- Agent prefers type-0 actions (existing boundary vertices), producing many small quads

## Cross-Domain Generalization

Star-trained model evaluated on unseen domains:
| Domain     | Vertices | Elements | Quads | Tris | Quality | Complete |
|------------|----------|----------|-------|------|---------|----------|
| L-shape    | 6        | 20       | 18    | 2    | 0.245   | Yes      |
| Octagon    | 8        | 16       | 16    | 0    | 0.392   | No       |
| Circle     | 16       | 100      | 100   | 0    | 0.367   | No       |
| Rectangle  | 20       | 100      | 100   | 0    | 0.304   | No       |

The agent generalizes to produce valid quads on all domains, but doesn't complete larger ones (100-step episode limit truncates). The L-shape completes because it has few vertices.

## SAC vs DQN

SAC with continuous actions fundamentally failed: valid action rate is <5% for random actions, so the agent never discovers valid quads during warmup. DQN with action masking solves this by only considering valid actions.

## Phase 6 Revision: Quality-Weighted Reward

Changes: 2x quality weight in reward, aspect ratio penalty (>3:1), max episode length 100->200.

### Revised Training (50k DQN on Star Domain)
| Timestep | Eval Return | Completion |
|----------|-------------|------------|
| 5,000    | 13.46       | 100%       |
| 10,000   | 34.53       | 100%       |
| 15,000   | 15.95       | 100%       |
| 20,000   | 15.29       | 0%         |
| 25,000   | 29.48       | 100%       |
| 30,000   | 21.30       | 100%       |
| 35,000   | 44.27       | 100%       |
| 40,000   | 31.58       | 100%       |
| 45,000   | 34.59       | 100%       |
| 50,000   | 37.03       | 100%       |

**Final eval: 55 elements (53Q+2T), quality=0.331, complete.**
Compared to original: more elements placed (55 vs 43), all-quad on L-shape (21Q vs 18Q+2T).
Aspect ratio penalty successfully penalizes extremely elongated quads.

## Key Weaknesses
1. **Quality**: Mean quality ~0.3-0.4 is mediocre. Many quads are elongated or skewed.
2. **Episode length**: 100-step limit insufficient for larger domains.
3. **Fixed domain**: Agent trains on one domain only; no domain randomization.
4. **Type-1 actions**: Agent rarely uses type-1 (new vertex placement), relying on type-0.

## Next Steps Plan

### Immediate (next session)
1. **Increase max episode length** to 200-300 for larger domains
2. **Domain randomization**: Randomize domain each episode from the 5 registered domains
3. **Reward shaping for quality**: Increase weight of equiangular quality in reward

### Short-term
4. **Test on fort_14 domains** from CHILmesh repo — 4 files available:
   - `annulus_200pts.fort.14` — annular domain, 200 pts
   - `donut_domain.fort.14` — donut/ring topology
   - `Block_O.14` — block O-grid
   - `structuredMesh1.14` — structured mesh
   All are ADCIRC fort.14 format (node coords + element connectivity).
5. **Write fort.14 parser**: Extract boundary polygons from ADCIRC mesh files for use as MeshEnvironment domains
6. **Type-1 action tuning**: Encourage new vertex placement for better quality

### Medium-term
7. **Post-processing**: Fix doublet_collapse bug and enable mesh cleanup
8. **Layer-based meshing**: Implement proper QuADMESH+ layer structure (Layers x FreeMeshRL)
9. **Curriculum learning**: Train on simple domains first, then harder ones
10. **Domain randomization**: Train on random domain each episode to improve generalization
