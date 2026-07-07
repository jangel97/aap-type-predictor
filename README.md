# AAP Type Predictor

Learned entity type prediction for Typed Composition Search (TCS), demonstrated on the Ansible Automation Platform MCP Server (1,060 tools, 79 entity types).

A frozen sentence encoder (all-MiniLM-L6-v2, 22M params) with a lightweight MLP head (~278K trainable parameters) predicts source and target entity types from natural language queries. BFS graph search then resolves the predicted types to tool chains.

## Results

Evaluated with 5-fold stratified cross-validation on 9,703 training examples, tested on 47 fixed held-out queries. Stratification by source type ensures minority entity types appear in every fold.

Four strategies are compared:

- **Function calling** — All 1,060 tools are passed as structured function definitions. The model uses its native tool-calling mechanism to select which functions to invoke.
- **Text baseline** — All 1,060 tools are listed as plain text in the system prompt (`- ToolName: (InputType) -> (OutputType)`). The model responds with a JSON array of tool names.
- **Zero-shot TCR** — Instead of tools, the model receives the 79 entity type names and predicts a source and target type. BFS graph search then resolves the types to a tool chain.
- **Encoder TCR** — Same as zero-shot TCR (type prediction + graph search), but type prediction is done by the trained 278K-parameter encoder instead of prompting an LLM.

For all strategies, precision, recall, and F1 are computed by comparing the predicted tool set against the expected tool set using set intersection. For TCR strategies (zero-shot and encoder), the model predicts two things — a source type and a target type — and both must be correct for the tool chain to resolve correctly.

| Strategy | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| Function calling | Gemini 2.5 Pro | 0.31 | 0.34 | 0.31 |
| Text baseline | Gemini 2.5 Pro | 0.56 | 0.87 | 0.66 |
| Zero-shot TCR | Gemini 2.5 Pro | 0.81 | 0.80 | 0.80 |
| **Encoder TCR** | **278K params** | **0.93 +/- 0.03** | **0.93 +/- 0.03** | **0.93 +/- 0.03** |

Per-category (mean across 5 folds):

| Category | N | Precision | Recall | F1 |
|---|---|---|---|---|
| clean | 15 | 0.96 | 0.96 | 0.96 |
| multihop | 10 | 0.99 | 0.98 | 0.98 |
| synonym | 7 | 0.97 | 0.97 | 0.97 |
| ambiguous | 5 | 0.74 | 0.80 | 0.76 |
| noisy | 5 | 0.76 | 0.76 | 0.76 |
| multipath | 5 | 1.00 | 1.00 | 1.00 |

### Gemini 2.5 Pro comparison

Gemini 2.5 Pro — a frontier model — was evaluated on the same 47 queries across all three strategies:

- **Function calling (F1=0.31)**: With 1,060 tools as function definitions, Gemini returns empty tool calls for most queries. The tool catalog overwhelms the model — it only manages to select tools for 16 of 47 queries.
- **Text baseline (F1=0.66)**: Gemini achieves high recall (0.87) by selecting multiple tools per query, but precision drops to 0.56 — it over-selects, returning 2-3 tools where one is expected.
- **Zero-shot TCR (F1=0.80)**: Gemini's best strategy. It correctly infers type relationships for 37 of 47 queries (79% exact match), but struggles with Platform-source queries — predicting self-referential types like `Activation->Activation` and `Credential->Credential` instead of `Platform->X`.

The 278K-parameter encoder beats Gemini 2.5 Pro on every strategy. The zero-shot TCR failure pattern is telling: Gemini gets 0% on synonym queries that map informal terms to AAP-specific types (e.g., "secrets" -> Credential, "event handlers" -> Activation). The encoder learned these mappings from domain-specific training data.

### Confidence threshold analysis

The model's confidence (min of source and target softmax max) enables selective prediction. At higher thresholds, the model abstains from low-confidence predictions, trading coverage for accuracy.

| Threshold | Coverage | P | R | F1 |
|---|---|---|---|---|
| 0.00 | 100% | 0.96 | 0.96 | 0.96 |
| 0.50 | 96% | 0.98 | 0.98 | 0.98 |
| 0.80 | 83% | 1.00 | 1.00 | 1.00 |
| 0.90 | 72% | 1.00 | 1.00 | 1.00 |

## Usage

```bash
pip install -r requirements.txt

# Train with 5-fold stratified CV (uses GPU if available)
python train.py --hidden-dim 512 --epochs 300

# Evaluate saved model on held-out test queries
python evaluate.py

# Run confidence threshold analysis
python evaluate.py --threshold-sweep

# Benchmark Gemini 2.5 Pro (requires GEMINI_API_KEY)
python benchmark_gemini.py
```

## Data

See [`data/README.md`](data/README.md) for detailed documentation of the training data construction process, style distribution, and key design decisions.

### Graph

`data/graph_snapshot.json` — Typed composition graph derived from 4 AAP OpenAPI specs (Controller, EDA, Galaxy, Gateway). Each tool is modeled as a typed transformation `input_types -> output_types`. Entity types are extracted from request/response schemas; tools that share types create edges in the graph.

- 79 entity types, 1,060 tools, 259 direct (source, target) pairs

### Training data

`data/training_data.jsonl` — 9,703 examples across 10 query styles (LLM paraphrases, deterministic templates, multihop paths, hard negatives, synonyms, and more). Source type distribution is flatter than Stripe — Platform accounts for 25% of examples (vs 57% on Stripe), with 48 unique source types.

### Test queries

`data/test_queries.json` — 47 held-out evaluation queries across 6 difficulty categories (clean, multihop, synonym, ambiguous, noisy, multipath). No overlap with training data.

## Architecture

```
Query (natural language)
    |
    v
all-MiniLM-L6-v2 (frozen, 384-dim)
    |
    v
Shared MLP: Linear(384, 512) + ReLU + Dropout(0.2)
    |
    +---> Source Head: Linear(512, 79) ---> 79 logits ---> argmax ---> source_type
    |
    +---> Target Head: Linear(512, 79) ---> 79 logits ---> argmax ---> target_type
                                                                           |
                                                        BFS Graph Search <-+
                                                              |
                                                              v
                                                         Tool Chain
```

Each head outputs 79 logits (one per entity type). Argmax selects the predicted type, softmax gives a confidence score. The minimum of source and target confidence is used as the overall prediction confidence. Both types must be correct for the tool chain to resolve correctly.

Training: 300 epochs per fold, cosine LR schedule (1e-3 -> 1e-5), sqrt-scaled class weights for source/target imbalance. Best model (by test F1) is saved.

## Comparison with Stripe

This is a companion repo to [stripe-type-predictor](https://github.com/jmorenas/stripe-type-predictor), which demonstrates the same architecture on the Stripe API (536 tools, 163 entity types, F1=0.91). Together they show that Typed Composition Search generalizes across domains:

| | Stripe | AAP |
|---|---|---|
| Tools | 536 | 1,060 |
| Entity types | 163 | 79 |
| Training examples | 3,388 | 9,703 |
| Encoder F1 | 0.91 +/- 0.02 | 0.93 +/- 0.03 |
| Best Gemini strategy | Text (0.82) | Zero-shot TCR (0.80) |
| Encoder params | 182K | 278K |

As the tool count doubles (536 -> 1,060), Gemini's function calling degrades sharply (0.40 -> 0.31) while the encoder maintains high accuracy. The type prediction reformulation reduces the search space from thousands of tools to tens of entity types, making the problem tractable for a lightweight classifier.
