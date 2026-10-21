# Vector databases

Source for this question. The problem is the [note](../README.md). The page is https://kpvarma5899.github.io/AiLlmConcepts/vector-databases/.

The Cursor canvas that this page was written from is [`canvas.tsx`](../canvas.tsx). It is the same note: distance math, the three charts, the build path, and the fifteen probes. GitHub Pages serves the rendered note. The canvas opens beside the editor.

No service is invented here. These are the types you name on the board, and the pattern each one is holding.

| Type | Holds | Pattern |
|---|---|---|
| `Chunk` | Normalized text, parent section id, content hash, tenant id | Structure-aware split. The chunk table is the source of truth. |
| `Embedding` | Unit vector, model version, dimension, metric | Pure function of text plus model. A new model is a new collection. |
| `Neighbor` | Chunk id, distance, payload | ANN candidate. Not a proof of the true nearest neighbor, and not an authorization decision. |
| `FusionScore` | Reciprocal rank from the vector list and the BM25 list | `1 / (60 + rank)`. Ranks fuse. Raw cosine and BM25 scores do not. |
| `Rerank` | Cross-encoder score over the fused pool of 50–100 | Repairs the top of a short list. Timeout falls back to fusion order. |

The index is a derived structure. Rebuild it from `Chunk` when the model changes or the HNSW dead-node fraction crosses the threshold you measured.
