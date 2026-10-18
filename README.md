# Sem-SAEA

Implementation of **Data-Adaptive Semantic Priors for Surrogate-Assisted Evolutionary Optimization of Expensive Constrained Multi-Objective Problems**. Response-specific metadata set ARD length-scale prior centres; archive evidence gates those priors before feasibility-aware allocation spends the next exact evaluation.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

`configs/zdt1.json` defines the ten-variable analytic pymoo ZDT1 evaluator and response/variable descriptions. Preparation caches frozen semantic relations from `sentence-transformers/all-MiniLM-L6-v2` (downloaded on first use):

```bash
semsaea prepare --config configs/zdt1.json --output data/prepared --seed 2026001
```

## Training and inference

```bash
semsaea run --data data/prepared --budget 300 --seed 2026001 --output runs/full
```

## Evaluation

The run records exact evaluator calls and their feasible nondominated set. `--front reference.csv` enables IGD+ against a supplied reference front.

## Comparisons and ablations

```bash
semsaea prepare --config configs/zdt1.json --blank --output data/numerical --seed 2026001
semsaea run --data data/numerical --budget 300 --seed 2026001 --output runs/numerical
semsaea prepare --config configs/zdt1.json --shuffled --output data/shuffled --seed 2026001
semsaea run --data data/shuffled --budget 300 --seed 2026001 --output runs/shuffled
semsaea run --data data/prepared --budget 300 --seed 2026001 --no-gate --output runs/no_gate
semsaea run --data data/prepared --budget 300 --seed 2026001 --no-decay --output runs/no_decay
```

## Outputs

Prepared inputs contain `problem.json` and `semantics.npz`. Runs write `archive.npz`, `history.json`, `metrics.json` and `pareto.png`. Engineering inputs use `evaluator: "module:function"`: a callable from one physical decision vector to finite objective/constraint arrays; constraints <= 0 are feasible.
