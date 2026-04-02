# From Search to Computation: Structured Error Tracing

This repository contains code for the paper:

**From Search to Computation: Structured Error Tracing via Recursive Luoshu Localization**

Error tracing in neural networks is typically treated as a search problem over internal states. In this work, we study a different regime in which tracing becomes direct computation under structured localization.

We compare three tracing regimes:

* **A0 (Search):** exhaustive scanning over candidate locations
* **A1 (Guided Search):** anchor-based reduction of the search space
* **A2 (Computation):** direct coordinate decoding via anchor–path structure

The central result is a transition from search-based behavior to structured computation.

Localization is treated here not just as a saliency signal, but as a representation that determines how tracing is performed. Anchor signals provide coarse regions, while path structure encodes positional identity. Only their combination enables exact coordinate recovery without search.

## Code structure

* `run_experiment.py` — entry point for running experiments
* `model_setup.py` — model and structured localization setup
* `tracing_cost_analysis.py` — tracing cost measurement and analysis

## Reproducing results

Run:

```
python run_experiment.py
```

This will reproduce the tracing cost comparison across A0, A1, and A2.

## Notes

All A1 results are reported under a unified guided-search protocol with a fixed tracing-cost definition. Absolute cost may vary across implementations, but the qualitative distinction between search and computation remains unchanged.

