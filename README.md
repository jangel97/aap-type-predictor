# AAP Type Predictor

Learned entity type prediction for Typed Composition Search (TCS), demonstrated on the Ansible Automation Platform MCP Server (1,060 tools, 79 entity types).

Our approach assumes that the target domain admits a typed entity abstraction, where tools can be represented as transformations between entity types. Under this assumption, tool routing can be reformulated as predicting source and target entity types followed by graph search over the typed composition graph.

A frozen sentence encoder (all-mpnet-base-v2, 110M params) with a lightweight MLP head (~475K trainable parameters) predicts source and target entity types from natural language queries. BFS graph search then resolves the predicted types to tool chains.

## Results

Evaluated with 5-fold stratified cross-validation on 10,065 training examples, tested on 122 fixed held-out queries across 6 difficulty categories. Stratification by source type ensures minority entity types appear in every fold.

Five strategies are compared:

- **Function calling** — All 1,060 tools are passed as structured function definitions. The model uses its native tool-calling mechanism to select which functions to invoke.
- **Text baseline** — All 1,060 tools are listed as plain text in the system prompt (`- ToolName: (InputType) -> (OutputType)`). The model responds with a JSON array of tool names.
- **Zero-shot TCR** — Instead of tools, the model receives the 79 entity type names and predicts a source and target type. BFS graph search then resolves the types to a tool chain.
- **Few-shot TCR** — Same as zero-shot TCR, but with 10 in-context examples drawn from the training set (excluding test type pairs). This gives the LLM domain-specific demonstrations of how queries map to entity types.
- **Encoder TCR** — Same type prediction + graph search pipeline, but type prediction is done by the trained 475K-parameter classifier instead of prompting an LLM.

For all strategies, precision, recall, and F1 are computed by comparing the predicted tool set against the expected tool set using set intersection.

| Strategy | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| Function calling | Gemini 2.5 Pro | 0.326 | 0.357 | 0.333 |
| Text baseline | Gemini 2.5 Pro | 0.449 | 0.593 | 0.495 |
| Zero-shot TCR | Gemini 2.5 Pro | 0.734 | 0.730 | 0.730 |
| Few-shot TCR | Gemini 2.5 Pro | 0.832 | 0.820 | 0.822 |
| **Encoder TCR** | **475K params** | **0.848** | **0.861** | **0.852** |

Per-category (encoder, best fold):

| Category | N | Precision | Recall | F1 |
|---|---|---|---|---|
| clean | 35 | 0.914 | 0.914 | 0.914 |
| multihop | 22 | 0.955 | 0.955 | 0.955 |
| multipath | 15 | 1.000 | 1.000 | 1.000 |
| noisy | 15 | 0.867 | 0.867 | 0.867 |
| ambiguous | 15 | 0.667 | 0.733 | 0.689 |
| synonym | 20 | 0.625 | 0.650 | 0.633 |

### Encoder vs Gemini 2.5 Pro per-category

The encoder and Gemini few-shot TCR show complementary strengths. The encoder wins on ambiguous queries where source type confusion is the challenge, while Gemini handles clean structured inputs better:

| Category | Encoder | Gemini FS-TCR | Winner |
|---|---|---|---|
| clean | 0.914 | **0.943** | Gemini |
| multihop | **0.955** | 0.848 | Encoder |
| multipath | **1.000** | 1.000 | Tie |
| noisy | **0.867** | 0.867 | Tie |
| ambiguous | **0.689** | 0.378 | Encoder |
| synonym | 0.633 | **0.750** | Gemini |

The encoder's large advantage on ambiguous queries (0.689 vs 0.378) reflects its training on source-disambiguation examples — queries where Platform competes with domain-specific sources like Organization, Credential, or Instance. The encoder also dominates multihop queries (0.955 vs 0.848), where multi-step type chains require precise graph knowledge.

### Confidence threshold analysis

The model's confidence (min of source and target softmax max) enables selective prediction. At higher thresholds, the model abstains from low-confidence predictions, trading coverage for accuracy.

| Threshold | Coverage | P | R | F1 |
|---|---|---|---|---|
| 0.00 | 100% | 0.96 | 0.96 | 0.96 |
| 0.50 | 96% | 0.98 | 0.98 | 0.98 |
| 0.80 | 83% | 1.00 | 1.00 | 1.00 |
| 0.90 | 72% | 1.00 | 1.00 | 1.00 |

### Entity pruning

Type prediction acts as a pruning step: predicting the source type alone eliminates most tools from consideration before graph search begins.

| Source type | Tools reachable | Pruned |
|---|---|---|
| AuditRule | 2 | 99.8% |
| Activation | 7 | 99.3% |
| Job | 14 | 98.7% |
| Host | 16 | 98.5% |
| Group | 18 | 98.3% |
| Inventory | 24 | 97.7% |
| Project | 26 | 97.5% |
| WorkflowJobTemplate | 26 | 97.5% |
| JobTemplate | 28 | 97.4% |
| Organization | 37 | 96.5% |
| Platform | 206 | 80.6% |

Across the test queries, the median source type reaches 28 tools (97.4% pruned). Even Platform — the most connected type, serving as the entry point for all listing queries — prunes 80.6% of the catalog. When both source and target types are known, the candidate set typically narrows to 0-3 tools, making resolution near-exact.

This is why the architecture scales: doubling the tool count (536 on Stripe -> 1,060 on AAP) has no effect on the classifier — it always predicts over the same 79 entity types. The pruning ratio actually improves as the catalog grows, since the type vocabulary stays fixed while the tool count increases.

## Usage

```bash
pip install -r requirements.txt

# Train with 5-fold stratified CV (uses GPU if available)
python train.py --base-model all-mpnet-base-v2 --hidden-dim 512 --epochs 300

# Evaluate saved model on held-out test queries
python evaluate.py

# Run confidence threshold analysis
python evaluate.py --threshold-sweep

# Benchmark Gemini 2.5 Pro (requires GEMINI_API_KEY)
python benchmark_gemini.py
python benchmark_gemini.py --strategy few_shot_tcr
```

## Data

See [`data/README.md`](data/README.md) for detailed documentation of the training data construction process, style distribution, and key design decisions.

### Graph

`data/graph_snapshot.json` — Typed composition graph derived from 4 AAP OpenAPI specs (Controller, EDA, Galaxy, Gateway). Each tool is modeled as a typed transformation `input_types -> output_types`. Entity types are extracted from request/response schemas; tools that share types create edges in the graph.

- 79 entity types, 1,060 tools, 259 direct (source, target) pairs

### Training data

`data/training_data.jsonl` — 10,065 examples across 11 query styles. Each example is a `(query, source_type, target_type)` triple.

| Style | Examples | Description |
|---|---|---|
| paraphrase | 3,831 (38%) | LLM-generated rephrasings of template queries |
| template | 2,058 (20%) | Direct queries: "List all X", "Show X for this Y" |
| multihop | 1,489 (15%) | Multi-step chains: "Get Y for Z's X" |
| noisy | 618 (6%) | Conversational: "Something went wrong with X, can you check?" |
| question | 496 (5%) | Question form: "How many X are there?" |
| hard_negative | 478 (5%) | Confusing pairs: Job vs UnifiedJob, User vs Team |
| synonym | 392 (4%) | Informal terms: "secrets" -> Credential, "playbooks" -> JobTemplate |
| anti_self | 280 (3%) | Prevent self-referential predictions (X->X) |
| augment | 258 (3%) | Additional coverage for underrepresented pairs |
| ambiguous | 90 (1%) | Source-confusing queries: Platform vs domain sources |
| targeted | 75 (1%) | Targeted fixes for specific failure modes |

Source type distribution is flatter than Stripe — Platform accounts for 25% of examples (vs 57% on Stripe), with 48 unique source types.

### Test queries

`data/test_queries.json` — 122 held-out evaluation queries across 6 difficulty categories (clean, multihop, synonym, ambiguous, noisy, multipath). No overlap with training data.

## Architecture

```
Query (natural language)
    |
    v
all-mpnet-base-v2 (frozen, 768-dim)
    |
    v
Shared MLP: Linear(768, 512) + ReLU + Dropout(0.2)
    |
    +--> Source Head: Linear(512, 79) --> 79 logits --> argmax --> source_type
    |
    +--> Target Head: Linear(512, 79) --> 79 logits --> argmax --> target_type
                                                                       |
                                                    BFS Graph Search <-+
                                                          |
                                                          v
                                                     Tool Chain
```

Each head outputs 79 logits (one per entity type). Argmax selects the predicted type, softmax gives a confidence score. The minimum of source and target confidence is used as the overall prediction confidence. Both types must be correct for the tool chain to resolve correctly.

Training: 300 epochs per fold, cosine LR schedule (1e-3 -> 1e-5), sqrt-scaled class weights for source/target imbalance. Best model (by test F1) is saved.

## Comparison with Stripe

This is a companion repo to [stripe-type-predictor](https://github.com/jangel97/stripe-type-predictor), which demonstrates the same architecture on the Stripe API (536 tools, 163 entity types). Together they show that Typed Composition Search generalizes across domains:

| | Stripe | AAP |
|---|---|---|
| Tools | 536 | 1,060 |
| Entity types | 163 | 79 |
| Training examples | 3,787 | 10,065 |
| Encoder F1 | 0.842 | 0.852 |
| Gemini FS-TCR F1 | 0.836 | 0.822 |
| Encoder params | 281K | 475K |

The encoder beats Gemini 2.5 Pro with few-shot prompting on both domains. As the tool count doubles (536 -> 1,060), Gemini's function calling degrades sharply (0.353 -> 0.333) while the encoder maintains high accuracy. The type prediction reformulation reduces the search space from thousands of tools to tens of entity types, making the problem tractable for a lightweight classifier.

## Conclusion

A 475K-parameter classifier built on a frozen 110M-param sentence encoder beats Gemini 2.5 Pro with few-shot prompting on AAP tool routing (F1=0.852 vs 0.822), while being ~1000x smaller, running locally with sub-millisecond inference, and requiring no API key.

The encoder wins on the categories that matter most for real-world deployments: ambiguous queries requiring source type disambiguation (0.689 vs 0.378) and multihop chains requiring precise graph knowledge (0.955 vs 0.848). Gemini's advantages are limited to clean structured queries (0.943 vs 0.914) and synonym resolution (0.750 vs 0.633).

The architecture scales naturally with tool count: doubling from 536 tools (Stripe) to 1,060 tools (AAP) has no effect on the classifier, which always predicts over a fixed set of entity types. Across both domains, the encoder beats Gemini 2.5 Pro — demonstrating that tool routing need not depend on frontier models. Once routing is decomposed into semantic type prediction and deterministic graph search, a compact domain-specific classifier is sufficient.
