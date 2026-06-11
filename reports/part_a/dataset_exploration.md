# Part A — Dataset Exploration (GLB internal structure)

14 `.glb` glasses assets. Each loads via `trimesh` as a **Scene** of one or more mesh components (frame / lenses / hinges …) with PBR materials. The table reports each file's internal organisation; the observations below motivate two pipeline decisions documented in the README.

| asset | components | vertices | faces | materials | textured | main colours | extent (w×h×d) |
|---|---|---|---|---|---|---|---|
| 00686245121504 | 4 | 17,890 | 32,960 | 4 | yes | #666666 | 0.152×0.038×0.149 |
| 00686245122204 | 4 | 17,890 | 32,960 | 4 | yes | #666666 | 0.147×0.038×0.149 |
| 00712316925280 (1) | 4 | 32,360 | 59,344 | 4 | yes | #666666 | 0.148×0.043×0.161 |
| 00712316925297 | 4 | 31,926 | 59,344 | 4 | yes | #666666 | 0.148×0.043×0.161 |
| 00800414475704 | 5 | 49,019 | 81,840 | 5 | yes | #666666 | 0.15×0.035×0.156 |
| 00800414475711 | 5 | 48,095 | 82,664 | 5 | yes | #666666 | 0.152×0.035×0.151 |
| 00800414475742 | 4 | 49,952 | 91,680 | 4 | yes | #666666 | 0.154×0.041×0.152 |
| 00800414526970 | 4 | 37,688 | 57,002 | 4 | yes | #666666 | 0.147×0.046×0.146 |
| 00800414559046 | 6 | 18,674 | 32,288 | 6 | yes | #666666, #694e4f | 0.137×0.047×0.147 |
| 00800414568628 | 7 | 54,667 | 94,231 | 7 | yes | #666666 | 0.127×0.029×0.136 |
| 00803926422709 | 4 | 21,316 | 31,244 | 4 | yes | #666666 | 0.149×0.046×0.146 |
| 00803926422716 | 4 | 20,774 | 29,964 | 4 | yes | #666666 | 0.149×0.046×0.148 |
| 00805289304449 | 5 | 28,865 | 53,168 | 5 | yes | #666666 | 0.157×0.047×0.146 |
| 00805289304456 | 5 | 38,458 | 70,772 | 5 | yes | #666666 | 0.148×0.047×0.145 |

**Observations**
- Mesh components per asset range **4–7** — most GLBs are multi-component Scenes, so flattening must **apply each node's scene-graph transform** (`Scene.dump(concatenate=True)`), not merge in local frames (see README → Challenges).
- Vertex counts span **17,890–54,667**; we sample a fixed 1024 surface points for the 3D (Point-MAE) feature so geometry detail is comparable across assets.
- **14/14** assets carry texture/material colour. That colour is real (baked into the coloured *hover* renders) but the encoders embed **greyscale** shape renders / pure xyz geometry — colour is intentionally **not** a clustering signal.
