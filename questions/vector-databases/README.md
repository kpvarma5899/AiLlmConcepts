# Vector databases and the math under LLMs

**Page:** https://kpvarma5899.github.io/AiLlmConcepts/vector-databases/

A vector database does not understand text. It stores points produced by an embedding model and returns the nearest ones under a distance you chose. The LLM that answers the user is a different model, trained with a different loss. Most production failures come from treating those two systems as one.

**Close on this:** if both vectors have length 1, cosine similarity, dot product, and Euclidean distance produce the same ranking. The identity is ‖a − b‖² = 2 − 2(a · b). They disagree only when magnitude is allowed to vary, or when the two vectors came from different models.

| | Retrieval | Generation |
|---|---|---|
| You store | One vector per chunk, from a contrastive model | Nothing from the chat model. It reads the chunks you retrieved. |
| You defend | The metric, the index, the filter, and hybrid fusion | Attention, softmax, and why a last hidden state is a bad embedding |
| They are testing | Whether the neighbor is the nearest legal point in the right model space | Whether you can separate a retrieval miss from a hallucination |

| | |
|---|---|
| Vector bytes, 50M × 1536 float32 | 286 GiB |
| Cosine, dot, and L2 on unit vectors | 1 ranking |
| Reciprocal rank fusion constant | k = 60 |
| KV cache per token, worked example | 128 KiB |

## What actually runs on a query

Retrieval is a funnel. Each stage is allowed to be wrong in a different way, and only the last stage is allowed to be slow.

| Stage | What it does | Budget | What breaks if you skip the discipline |
|---|---|---|---|
| Embed | Same model, same pooling, same query or passage prefix, same normalization as ingest. A pure function of text plus model version. | One batched call | You search a different space than the one you indexed |
| ANN | HNSW or IVF returns a candidate pool, not a proof of the true nearest neighbors | About 50–100 hits | You either scan N·d or you trust a guess as the final order |
| Fuse and filter | Merge with BM25 by reciprocal rank fusion. Authorization is a predicate on the index. | One lexical query plus the walk | Exact codes miss, or another tenant's chunk is a neighbor |
| Rerank | A cross-encoder reads query and chunk together and keeps 5–8 | Those chunks enter the prompt | A pooled vector cannot represent negation or a rare token |

## Distance math

An embedding is a point in R^d. Typical retrieval dimensions are 384, 768, 1024, 1536, and 3072. The database's job is arg-sort by a similarity, then apply metadata predicates. Exact search computes the metric against all N points: cost Θ(N · d) per query. At 10 million vectors of dimension 1536 that is about 1.5×10^10 multiply-adds before you have done any filtering. Approximate nearest neighbor search exists because that number does not fit an interactive budget.

### Three metrics, and when they are the same

```
a · b = Σᵢ aᵢ bᵢ
cos(a, b) = (a · b) / (‖a‖ ‖b‖)
‖a − b‖² = ‖a‖² + ‖b‖² − 2(a · b)
```

Substitute ‖a‖ = ‖b‖ = 1 into the third line and the first two terms become 1 + 1. Squared L2 distance is then exactly 2 − 2(a · b). Minimizing L2, maximizing the dot product, and maximizing cosine are the same sort.

Leave the vectors unnormalized and the dot product rewards length. A long, slightly-off vector beats a short, well-aligned one. Cosine throws length away. L2 treats length as part of the meaning. Use the metric the embedding model was trained with. For retrieval models that is cosine, implemented as a dot product after L2 normalization.

### Worked example

a = (1, 0), already unit length. b = (0.6, 0.8), and 0.6² + 0.8² = 1, so b is unit length too.

a · b = 0.6. Cosine = 0.6. ‖a − b‖² = 0.4² + (−0.8)² = 0.16 + 0.64 = 0.80, which equals 2 − 2×0.6. Any of the three metrics ranks b the same against every other unit vector.

Stretch b to (1.2, 1.6), which is the same direction at length 2. Cosine stays 0.6. The dot product becomes 1.2 and would now outrank a unit vector whose cosine is 0.9. That single stretch is the bug behind "cosine and inner product returned different neighbors."

### What the coordinates mean

Nothing in the database assigns a human label to dimension 17. The coordinates are whatever basis the embedding model's last projection learned. Distance is meaningful only inside one basis: same weights, same dimension, same pooling (mean, CLS, or last token), same normalization, and the same task prefix. E5-style models prepend "query: " and "passage: ". Drop the prefix and you are searching a slightly different space than the one the loss trained.

High dimension does not by itself make nearest neighbors meaningless. For points sprinkled uniformly in R^d, the nearest and farthest distances converge, which is the curse of dimensionality. Embedding sets are not uniform. They lie on a much lower-dimensional manifold of topics, so local neighborhoods stay informative. When they stop being informative you usually have anisotropy or hubness: the model packed every vector into a narrow cone, and a few hub vectors are the nearest neighbor of almost every query. Contrastive training, and sometimes mean-centering, is what pulls that cone open. Taking the raw last hidden state of a chat model does the opposite.

| Metric | Rank by | Use when | Breaks when |
|---|---|---|---|
| Cosine | Angle only | Retrieval embeddings trained with cosine or a normalized dot product | You needed magnitude, or only one side was normalized |
| Dot product | Angle and length | Both sides are unit vectors, or magnitude is a trained confidence | A long vector from a different encoder dominates |
| Euclidean | Straight-line gap | The model was trained with L2, or you already normalized and want the identity above | You compare it naively to a cosine score from another system |

pgvector operators: `<->` is L2, `<#>` is negative inner product (ascending order maximizes the dot product), `<=>` is cosine distance, which is 1 − cosine similarity.

## Indexes

Every ANN index is a bet that a structured subset of the distance computations still contains the true neighbors. You pay that bet in recall. The honest way to tune is a held-out query set whose ground truth came from a flat scan, then a curve of recall against latency. Vibes are not a recall number.

### HNSW, the default graph

Hierarchical Navigable Small World is a stack of graphs. The top layer has few nodes and long links. Each lower layer is denser. Layer 0 contains every vector. A query enters at the top and greedily walks to the neighbor closest to the query, then drops a layer and repeats. On layer 0 the walk keeps a candidate list of width `ef_search` instead of a single current node.

Two build knobs and one query knob matter. `M` is the maximum number of edges per node, often 16 to 48. `ef_construction` is the candidate width used while inserting; it decides whether the right edges exist at all. `ef_search` is the candidate width at query time. Raising `ef_search` spends latency to inspect more of the graph you already built. It cannot repair a graph built with a tiny `ef_construction`. Inserts are natural. Deletes are tombstones. The small-world edges are not rewired, so after enough updates the walk falls into dead neighborhoods and recall decays until you rebuild.

### Recall against latency as ef_search grows

<!-- widget:recall-ef -->

<!-- widget:latency-ef -->

### IVF and product quantization

An inverted file runs k-means, stores each vector in the list of its nearest centroid, and at query time scans only `nprobe` lists. `nprobe` = 1 is a hard Voronoi assignment: a true neighbor that sits just across a cell boundary is invisible, not merely ranked low. Raise `nprobe` the way you raise `ef_search`, and retrain centroids when the corpus distribution shifts, because the cells were fit to the training sample.

Product quantization compresses the vector. Split d dimensions into m subvectors. Each subspace gets a codebook, usually 256 centroids, so each subvector becomes one byte. A 768-d float32 vector is 3,072 bytes; m = 96 codes are 96 bytes. Distance to a query uses asymmetric distance computation: the query stays full precision, and you precompute a table of distances from each query subvector to the 256 centroids. A database distance is then m table lookups and an add. The estimate is good enough to propose candidates and bad enough to be the final order. Rescore the pool in float32, or at least int8.

Scalar int8 quantization is the compression to reach for first. It is about 4× on the vector bytes and usually loses little recall after a float rescore. Binary quantization keeps a sign bit per dimension and ranks with Hamming distance. It needs a wide candidate pool and a float rescore. It is a memory tool, not a quality tool.

### Vector bytes for 50 million × 1536-d

<!-- widget:storage-50m -->

| Structure | What it stores | Query knob | Failure you will actually hit |
|---|---|---|---|
| Flat | The vectors | None. Exact. | Latency. This is the ground truth you score ANN against, not the production path past about 10^5–10^6 vectors. |
| HNSW | Vectors plus a multi-layer graph | `ef_search` | Tombstones after deletes. Recall quietly decays. Filtered walks that never enter the allowed region. |
| IVF | Centroid id plus an inverted list | `nprobe` | A neighbor across a Voronoi boundary. Stale centroids after the corpus drifts. |
| IVF-PQ | A coarse list id plus subspace codes | `nprobe`, then rescore depth | Trusting code distances as the final rank. The right chunk sits at position 40. |

**Filtered search is the actual hard problem.** Post-filtering runs ANN on the whole corpus, then drops rows that fail the ACL. If the predicate matches 0.2% of rows, a top-10 from the global graph is almost entirely illegal, and the survivors are not the nearest legal points. The walk has to know the predicate, or each tenant needs its own graph. Over-fetching is a patch you measure, not an architecture.

## LLM math

A generative LLM and an embedding model share a transformer skeleton and almost nothing else about the objective. The generator is trained to put probability on the next token. The embedder is trained to put matching passages near each other and non-matching passages far apart. The vector database only works if you stored the second kind of vector.

### One forward pass of a generator

Tokens become rows of a matrix. A learned table maps each token id to a vector, and a position signal is applied. Modern models use rotary position embeddings: pairs of coordinates in the query and key are rotated by an angle that depends on the token position, so the later dot product depends on relative distance. There is no separate position vector added at the end.

```
Q = X W_Q,   K = X W_K,   V = X W_V
Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V
P(i) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
```

If the entries of q and k are independent with mean 0 and variance 1, their dot product has variance dₖ. For dₖ = 128 that spread is wide. Softmax of a wide logit vector collapses toward a one-hot vector: one token gets nearly all the mass, the gradient through the others dies. Dividing by √dₖ puts the variance back near 1 before the exponential. That is the entire reason for the scale. It is not a tuning knob you set per query.

Multi-head attention runs several of these maps, concatenates them, and multiplies by an output matrix W_O. A residual add puts the result back onto the stream, and a feed-forward block mixes channels. Recent models use a gated form, SwiGLU: the hidden activation is a product of an up-projection and a gated up-projection, then a down-projection. The residual path is why a representation can survive many layers instead of being overwritten.

### Softmax temperature, numeric

Logits z = (2, 1, 0). At T = 1 the exponentials are about 7.39, 2.72, and 1.00. Divide by their sum, about 11.11, and the distribution is about (0.665, 0.245, 0.090).

At T = 0.5 the model divides the logits by 0.5 first, so the inputs to exp are (4, 2, 0). The distribution sharpens to about (0.867, 0.117, 0.016). As T approaches 0, sampling approaches argmax. As T grows, it approaches uniform. Top-p then truncates the tail and renormalizes. None of this changes the embedding geometry of a retrieved chunk. Temperature is a decoding choice on the generator, after retrieval.

### KV cache, because latency math is memory math

Autoregressive decoding recomputes attention over every past token unless K and V are stored. Per token, in bytes:

```
2 × n_layers × n_kv_heads × d_head × bytes_per_element
```

Worked example, not a specific product model: 32 layers, 8 key-value heads, head dimension 128, float16. That is 2 × 32 × 8 × 128 × 2 = 131,072 bytes, exactly 128 KiB per token. An 8,192-token sequence is 1 GiB of cache for that one request, before activations and before the weights. This is why serving is often memory-bandwidth bound, why continuous batching exists, and why a long-context "just stuff the PDF" design competes with retrieval on cost even when the model could physically accept the tokens.

### The loss that actually trains an embedding

A bi-encoder runs the query through one tower and the passage through another (often shared weights) and scores them with a dot product. Training uses a contrastive loss. In-batch negatives mean every other passage in the batch is a negative for this query.

```
L = −log[ exp(s(q, d⁺) / τ) / Σⱼ exp(s(q, dⱼ) / τ) ]
```

s is the similarity, usually a dot product of normalized vectors. τ is a temperature, but it is not the decoding temperature above. Here τ sharpens or softens the training distribution over negatives. Too large and the model under-separates neighbors. Too small and it overfits the hardest negative in the batch and ignores structure. The denominator is why batch size and hard-negative mining change the embedding space more than another epoch on easy pairs.

A cross-encoder does not produce a stored vector. It concatenates query and passage and emits one score, so token-level attention can see both sides. That is more accurate and costs a forward pass per pair, which is why it reranks 50 candidates and does not index 50 million chunks. A late-interaction model such as ColBERT sits between them: it stores a vector per token and scores with MaxSim, a sum of each query token's maximum dot product against the passage tokens. Storage grows with passage length. Recall on exact phrases gets better. The operational cost is multi-vector indexing, not a single HNSW.

| Object | Objective | What you store | What goes wrong if you mix them up |
|---|---|---|---|
| Decoder LLM | Next-token cross-entropy | Nothing in the vector index. The KV cache is ephemeral. | Last-token states cluster by language, length, and chat boilerplate. Hubs appear. |
| Bi-encoder | InfoNCE / contrastive | One vector per chunk | Rare identifiers and negations are weakly represented. Needs lexical search and a reranker. |
| Cross-encoder | Pairwise relevance | No corpus vectors. Score top candidates only. | Put it on the unfiltered corpus and the latency graph becomes linear in N. |

**Temperature names two different dials.** Decoding temperature divides logits inside the generator softmax. Contrastive temperature τ divides similarities inside InfoNCE. Changing the first changes which words you sample. Changing the second changes which passages land next to each other. Neither dial repairs a bad chunking scheme.

## How to build it

Given a Postgres background, the default that is actually easiest to operate is Postgres with pgvector until the corpus or the filter pattern outgrows it. The vector index is a derived structure. The source of truth stays the chunks table, so you can re-embed when the model changes without reconstructing the product.

### Write path

Parse by structure, not by a blind character window. Target roughly 200–500 tokens, split on headings, paragraphs, and function boundaries, and overlap only so a sentence is not cut in half. Store the parent section id. Retrieve the small chunk, generate from the parent if the chunk is too thin to answer. Hash the normalized text. Embed on a hash miss or a model-version change, in batches, with backpressure against the embedding API. Upsert by chunk id. Record model version, dimension, metric, and content hash on the row.

A 2,000-token chunk embeds cleanly and retrieves poorly. Pooling averages every token into one point, so a chunk about three topics becomes the centroid of those topics: somewhat near many queries, specific to none.

### Query path

Embed the query with the same function you used at ingest. Run HNSW with an `ef_search` you chose from a recall curve, over-fetching about 50–100, and a lexical query in parallel. Fuse with reciprocal rank fusion: score(d) = Σᵢ 1 / (60 + rankᵢ(d)). The constant 60 keeps a single first-place hit from erasing the other list. Apply the tenant and ACL predicate inside the scan. Rerank the fused pool with a cross-encoder down to 5–8 passages, with a timeout that falls back to fusion order. Log query, retrieved ids, scores, and the model version. Without those logs you cannot tell a retrieval bug from a generation bug.

| Choice | Default | Move off it when |
|---|---|---|
| Store | Postgres + pgvector HNSW, cosine ops, vectors normalized in the application | Filter-aware QPS no longer fits one primary, or you pass a few tens of millions of vectors and the working set leaves RAM. Qdrant is the usual next step because payload filters are part of the HNSW walk. |
| Library vs database | A database if a request must combine vectors, SQL predicates, and transactions | FAISS when you own persistence, filtering, and replication yourself and you need a specific index the database does not expose. FAISS is not a database. |
| Compression | float32 while it fits, else int8 plus float rescore of the top candidates | PQ or binary only under a hard RAM cap, and only with a rescore stage. |
| Hybrid | BM25 + vector, fused with RRF k = 60, then a cross-encoder | You drop BM25 only if an eval set shows no gain on identifiers, names, and error codes. That eval set almost always shows a gain. |
| Tenancy | A mandatory tenant predicate, or a partition per tenant | A shared global graph with the filter left to the prompt. That is a data leak, not a retrieval strategy. |
| Model change | A new collection. Dual-read during the eval, then cut over. | Writing new-model vectors into the old collection. The spaces are not the same, even at equal dimension. |

### What you measure before you tune

| Metric | Question it answers | Ceiling |
|---|---|---|
| Recall@k vs a flat scan | Did the index lose true neighbors the metric would have returned? | The flat scan. If recall@10 against brute force is 0.99 and answers are still wrong, the index is not the bug. |
| Recall@k vs labeled relevant chunks | Did the embedding and the chunking capture what a human calls relevant? | A perfect index cannot beat a bad embedding. This number sits under the flat-scan number. |
| MRR and nDCG | Is the best chunk near rank 1, or buried at rank 40 where the reranker might still save it and the prompt will not? | Use them once recall exists. They punish the right chunk in position 30. |
| Faithfulness, given the retrieved text | Did the generator assert something those chunks do not support? | Independent of recall. Log the ids or this metric is undefined. |

**Semantic cache is a vector query with a dangerous threshold.** Caching an answer when the new query's nearest cached query has cosine above 0.98 is reasonable for paraphrases. Caching at 0.9 is how "not eligible for a refund" inherits the answer for "eligible for a refund." Topic dominates polarity in embedding space. Exclude negations, numbers, and entities, or do not cache those queries. Key the cache by tenant as well as by vector.

## Follow-ups

Each probe is the question, then the decision, the failure mode of the other choice, and the mechanism that makes the decision true.

### Cosine, dot product, and L2 returned different neighbors

Normalize both sides to unit length at ingest and at query, store the metric in the collection metadata, and rank by dot product. Confirm the embedding model was trained for cosine. If it was trained for raw inner product on unnormalized vectors, do not normalize; magnitude is then part of the score.

The wrong move is to switch operators until the demo looks better. A vector that was stored unnormalized will win a dot-product index by being long, and lose a cosine index to a shorter vector with a better angle. Half-normalized collections are not a blend of the two. They are a third, accidental metric.

For unit vectors, ‖a − b‖² = 2 − 2(a · b), and cosine equals the dot product. The three sorts match. Any gap you observe is either magnitude or a comparison across two embedding spaces.

### An ACL that matches 0.2% of rows returns an empty or irrelevant top-k

Push the predicate into the index walk, or give each tenant its own graph. If the legal set is only a few thousand vectors, brute-force that set and skip ANN. Measure candidate depth; do not guess a 100× over-fetch and call it done.

Post-filtering asks the global HNSW for its favorite points and then throws away the illegal ones. The points that remain are the best global points that happened to be legal. They are not the nearest legal points. With a 0.2% pass rate, top-10 is often empty.

Greedy graph search only visits nodes on a path from the entry point. Legal nodes that sit off that path are never proposed, so no later filter can rank them. Filter-aware HNSW refuses illegal edges during the walk. pgvector's iterative scan keeps walking until the LIMIT is filled after the predicate. A partition per tenant makes the entry point itself legal.

### HNSW recall fell two weeks after you started deleting and reinserting

Treat the graph as append-mostly. Soft-delete, track the dead-node fraction, and rebuild when it crosses a threshold you have measured, often around 10–20% depending on M and the update rate.

Raising `ef_search` feels like the same knob that fixed recall at launch. It only widens the walk on the graph you have. It does not replace edges that now point at tombstones.

Edges are chosen at insert time against the graph that existed then. Deletes do not rerun the neighbor-selection heuristic. The small-world shortcut structure thins out, the greedy walk gets trapped, and the recall-versus-`ef_search` curve flattens below the bar you set on day one.

### You switched embedding models and queried the old collection

One collection per model version and dimension. Dual-write or dual-read during an offline eval, cut traffic over, then drop the old collection. Re-embed from the chunk table, which is why the chunk table is the source of truth.

Same dimension is not the same space. A 1,536-d vector from model A and a 1,536-d vector from model B are coordinates in different bases. Their cosine is a number without a relevance meaning. Different dimensions are not even a legal dot product.

The embedding is the image of the text under a specific set of weights, pooling, normalization, and task prefix. Change any of those and the nearest-neighbor relation learned by InfoNCE no longer applies. There is no linear map you can fit cheaply after the fact that preserves ranks for a new model. Orthogonal Procrustes maps do not replace re-embedding.

### Vector search cannot find ERR-4031, and keyword search can

Run BM25 and the vector search together. Fuse with reciprocal rank fusion at k = 60. Keep the identifier-heavy cases in the retrieval eval set so a future model upgrade cannot delete this path quietly.

Pure vector search looks strong on paraphrases in a demo and fails the first time a user pastes an error code, a SKU, or a person's name. The two codes ERR-4031 and ERR-4032 land near each other and near the prose "authentication failed."

Contrastive training is rewarded for pulling paraphrases together, which smooths away rare strings. Subword tokenization breaks an unseen code into pieces shared with many other codes. Sparse lexical scoring gives the exact term a weight the geometry will not. RRF does not require the two score ranges to be calibrated. It only uses ranks, which is why you do not add a cosine to a BM25 score.

### ef_search is 16 and recall@10 against a flat scan is 0.60

Sweep `ef_search` and plot recall@10 and p95 latency against the flat scan. Raise `ef_search` until the candidate stage clears the bar you need before reranking, commonly 0.95 or better. Touch `M` and `ef_construction`, and pay for a rebuild, only if that curve plateaus too low.

Rebuilding with a larger `M` on day one, or pushing `ef_search` into the thousands, both skip the measurement. The first is expensive and may be unnecessary. The second makes the graph walk approach the cost of the flat scan you were trying to avoid.

`ef_search` is the beam width on layer 0 of an existing graph. `ef_construction` is what decided whether the true neighbor is even linked within the region the beam will explore. Recall saturates at the quality of those edges.

### Will 50 million 1536-d vectors fit in one HNSW?

Count the bytes before you pick a product. float32 vectors alone are 50×10^6 × 1536 × 4 = 3.072×10^11 bytes, about 286 GiB. int8 is about 71.5 GiB. A binary code is about 8.9 GiB. A 96-byte PQ code is about 4.5 GiB. HNSW links add roughly N × M × 8–16 bytes, about 6–13 GiB at M = 16. If the float32 working set fits, keep it. If not, int8 plus a float rescore of the top candidates. PQ only under a hard RAM cap, and always rescore.

Loading a PQ index and sorting the user-visible hits by ADC distance will look fast and read wrong. Approximate distances reorder the top of the list. The passage you needed is often inside the candidate pool and outside the top 5 you put in the prompt.

Codes throw away coordinates. Asymmetric distance computation estimates the true distance from a full-precision query to a codebook reconstruction. Estimation error is small relative to random pairs and large relative to the gap between rank 1 and rank 20, which is exactly the gap RAG cares about.

### The query "not eligible for a refund" retrieves "eligible for a refund"

Do not treat a high cosine as logical equivalence. Rerank with a cross-encoder that sees both strings. Keep lexical features for polarity-bearing queries. If you run a semantic cache, threshold it very high, exclude negation, numbers, and entities, and always key it by tenant.

A cache threshold of 0.9 feels conservative and still collides on negation. The cached answer is then confidently wrong, which is worse than a cache miss.

InfoNCE mostly arranges topical neighborhoods. The direction that encodes "not" is small next to the direction that encodes "refund." A single pooled vector has no residual place to keep a modifier once it has been averaged across tokens. A cross-encoder runs attention over the pair, so the contradiction can affect the score directly.

### Why is the chat model's last hidden state a bad document embedding?

Use a model whose training loss was contrastive or otherwise built for retrieval. If you adapt a generator, use the pooling and fine-tune that adaptation actually trained, and still evaluate recall against a bi-encoder baseline before you trust it.

Writing the final-layer state of an instruction-tuned chat model into the index produces neighbors that share language, length, and stock phrases ("Sure, here is…") rather than content. A few hub vectors become nearest neighbors of a huge fraction of queries.

Next-token cross-entropy wants hidden states in a narrow cone so the unembedding matrix can pick likely tokens. That anisotropy is useful for generation and harmful for cosine search. Nothing in that loss pushes unrelated passages apart. InfoNCE does, because unrelated passages sit in the denominator.

### IVF with nprobe = 1 misses a neighbor the flat scan finds immediately

Raise `nprobe` and sweep it against recall the same way you sweep `ef_search`. Retrain centroids when the data distribution moves. If you need high recall without a training pass and the data is updated online, prefer HNSW.

`nprobe` = 1 is the setting people ship because the latency looks excellent. The missed neighbor is not a ranking error you can rerank away. It was never in the scanned lists.

k-means assigns each point to one Voronoi cell. Points near a face are closer to some points in the next cell than to most points in their own. The inverted list is a hard partition, so the query never computes those distances. `nprobe` is the number of neighboring cells you are willing to admit might hold the answer.

### How do you know retrieval failed rather than the generator hallucinated?

Score them separately. Retrieval: recall@k, MRR, and nDCG against labeled chunk ids, with a flat scan as the index ceiling. Generation: faithfulness against the chunks that were actually placed in the prompt. Log ids and scores on every request.

One end-to-end LLM-as-judge number sends you into a week of prompt edits while the index recall sits at 0.4. It also hides the opposite bug, where the right chunk was in the prompt and the model ignored it.

The retriever is a ranking function. It can be scored with no decoder in the loop. The generator is a conditional distribution given the prompt you built. A missing id in that prompt and an unsupported sentence given a present id are different defects, and only the log line distinguishes them.

### Tenant A's question returns tenant B's chunk

Make the tenant predicate mandatory in the data-access layer, or physically partition the index by tenant. Add a negative test: tenant A's queries must return zero tenant B rows. Key every cache by tenant. Do not rely on the system prompt to "only use this customer's data."

A global HNSW plus a filter someone can forget, or a semantic cache keyed only on the query text, both leak. Two tenants asking "what is our refund policy?" is a cache collision and a nearest-neighbor collision.

ANN returns geometry. Nothing in the distance computation knows about authorization unless the predicate removed those edges or those vectors live in another index. The prompt is not a security boundary.

### Re-embedding on every save exhausts the embedding API

Hash the normalized chunk text. Embed only on a hash miss or a model-version bump. Batch, and apply backpressure. Keep raw chunks in the source table so a model change is a rebuild, not a crisis.

Embedding the whole document on every keystroke, including whitespace, turns a derived index into the dominant cost and still leaves you unable to rebuild if you did not keep the text.

The embedding is a pure function of model version and normalized text. Equal inputs must produce equal vectors, so a content hash is a correct idempotency key. Whitespace-only changes that you forgot to normalize will still miss the cache. Normalization is part of the function, not a courtesy.

### Adding a reranker moved p95 from 40 ms to 800 ms

Retrieve 50–100 with the bi-encoder and the lexical list. Rerank that set to 5–8. Put a timeout on the cross-encoder and fall back to fusion order. Do not rerank thousands, and do not put the cross-encoder in front of the index.

Calling the reranker on the raw corpus, or on a top-1000 with no timeout, makes one GPU queue the owner of your tail latency. The index was fast because it never read the text.

A cross-encoder is O(k) full forward passes over query-chunk pairs. A bi-encoder is one query embedding plus an ANN walk. The cross-encoder exists because a single pooled vector dropped token-level evidence. Its job is to repair the top of a short list, which is only affordable when k is small.

### Chunks of 2,000 tokens embed fine and answers stay vague

Chunk on document structure to roughly 200–500 tokens. Store the parent section. Retrieve the child, and if the child is too narrow to answer, expand to the parent in the prompt. Overlap a sentence, not a second copy of the whole window.

Large fixed windows are simple and they dilute every query. They also fill the context with a few giant passages, which is where lost-in-the-middle shows up: the model attends to the ends of the prompt and underuses the middle passage.

One vector is a pool. Mean pooling averages token directions. The more unrelated directions you average, the shorter the component aligned with any single query, so the chunk becomes a generic neighbor of many queries. Parent-child retrieval keeps the indexed point specific and the generated context complete.
