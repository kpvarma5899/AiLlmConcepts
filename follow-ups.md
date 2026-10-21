# Follow-ups

A follow-up is a question an interviewer asks after the diagram is on the board. Answer it the way a high-quality engineer would: the decision first, then what breaks if you choose the other way, then the mechanism and the number the design depends on.

Do not leave a menu of options without picking one. Do not define a term and stop.

## Vector databases

Keep every one of these in `questions/vector-databases/README.md`.

- **Cosine, dot product, and L2 disagree.** Normalize both sides when the model was trained for cosine, then rank by dot product. The identity is ‖a − b‖² = 2 − 2(a · b) for unit vectors. A gap is magnitude, or two different embedding spaces.
- **An ACL matches 0.2% of rows and top-k comes back empty.** Push the predicate into the walk, or partition by tenant. Post-filtering returns the best global points that happened to be legal.
- **HNSW recall fell after deletes.** Tombstones do not rewire edges. Rebuild when the dead-node fraction crosses the threshold you measured. `ef_search` only widens the walk.
- **You changed embedding models.** New collection. Re-embed from the chunk table. Same dimension is still a different basis.
- **ERR-4031 is invisible to vectors.** BM25 plus vector, fused with RRF at k = 60. Do not add a cosine to a BM25 score.
- **`ef_search` is 16 and recall@10 is 0.60.** Sweep against a flat scan. Raise `ef_search` until the candidate stage clears about 0.95. Rebuild `M` and `ef_construction` only if the curve plateaus.
- **50 million × 1536-d.** float32 is about 286 GiB of vectors. int8 is about 71.5 GiB. PQ only with a float rescore.
- **Negation retrieves its opposite.** Topic dominates polarity. Cross-encode the pair. A semantic cache at 0.9 returns the wrong answer with confidence. Key it by tenant.
- **Last hidden state of the chat model.** Next-token loss packs states into a cone. Store a contrastive embedding. InfoNCE is what pushes unrelated passages apart.
- **`nprobe` = 1 misses a flat-scan neighbor.** The neighbor was in the next Voronoi cell, so it was never scored. Sweep `nprobe`. Prefer HNSW when the corpus updates online.
- **Retrieval bug or hallucination?** Recall against labeled ids, with the flat scan as the index ceiling. Faithfulness only against chunks that entered the prompt. Log the ids.
- **Tenant A sees tenant B.** The predicate is mandatory in the data-access layer, or the index is physically partitioned. The prompt is not a security boundary.
- **Re-embedding melts the API budget.** Content-hash the normalized text. Embed on a hash miss or a model-version change. The chunk table remains the source of truth.
- **The reranker owns p95.** Cross-encode 50–100 down to 5–8, with a timeout that falls back to fusion order. The cost is O(k) forward passes.
- **2,000-token chunks answer vaguely.** Index 200–500 token children split on structure. Generate from the parent section. One vector is a pool, and pooling dilutes every topic in the window.
