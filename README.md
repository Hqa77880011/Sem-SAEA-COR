# Sem-SAEA

Implementation of **Data-Adaptive Semantic Priors for Surrogate-Assisted Evolutionary Optimization of Expensive Constrained Multi-Objective Problems**. Response-specific metadata set ARD length-scale prior centres; archive evidence gates those priors before feasibility-aware allocation spends the next exact evaluation.

## Install and run

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
semsaea prepare --config configs/zdt1.json --output data/prepared
semsaea run --data data/prepared --budget 300 --output runs/full
```

Preparation downloads the frozen `sentence-transformers/all-MiniLM-L6-v2` encoder when absent from its cache. Variable records and response records are encoded separately, so the response-by-variable relation matrix varies across coordinates and objectives. The prepared matrix is cached once; exact outcomes never fine-tune the sentence encoder.

The supplied configuration runs ten-variable ZDT1 through pymoo. It contains variable descriptions, objective descriptions, constraints, and fixed ideal/nadir/reference values. Other ZDT, DTLZ and WFG problems can use the same schema with their actual dimensions, objective count and analytical normalization. Keep metadata independent of observed outcomes and use the same descriptions across paired method comparisons.

## Engineering evaluators

A configuration may replace `problem` with `evaluator: "my_problem:evaluate"`, together with `lower`, `upper`, optional `integer_indices`, variable/response records and fixed normalization. The importable Python callable accepts one physical decision vector and returns `(objectives, constraints)`, both one-dimensional arrays; feasibility means every constraint is at most zero. It must perform the actual simulator or measured response lookup and return finite values. Integer coordinates are rounded before evaluation, and the realized coordinates enter the exact archive.

Only exact evaluator responses can create archive observations or certify final feasibility. Surrogate-only designs never enter the returned Pareto set. For the thermal/hydraulic case, configure objective normalization from the declared temperature and pumping-power units and connect the actual CFD evaluator. The optimizer does not fabricate engineering responses.

## Comparisons and parameters

```bash
semsaea prepare --blank --output data/numerical
semsaea run --data data/numerical --output runs/numerical
semsaea prepare --shuffled --output data/shuffled
semsaea run --data data/shuffled --output runs/shuffled
semsaea run --no-gate --output runs/no_gate
semsaea run --no-decay --output runs/no_decay
```

Use identical `--seed` and `--budget` for paired initial designs. The blank control disables the semantic prior and does not load an encoder. Shuffled semantics permute variable records once. No-gate retains the decaying prior regardless of predictive evidence; no-decay retains constant prior strength while preserving the evidence gate. The numerical control shares the same GP, initial design, RVEA candidate search and acquisition.

`--population` defaults to 100 reference directions and `--generations` to 20 virtual RVEA generations. These surrogate calls do not consume exact budget. The initial design uses `5d` points from the best minimum-distance candidate among ten Latin hypercubes. Each subsequent iteration makes exactly one evaluator call. `--front reference.csv` adds IGD+ against a fixed reference front in physical objective units; use the same front for every method. `--encoder` selects an explicitly supplied compatible frozen sentence model during preparation.

## Outputs and metrics

`archive.npz` contains realized unit-domain decisions, exact objectives/constraints and feasible nondominated indices. `history.json` records exact-evaluation count, hypervolume, each response's gate state, LOO predictive density, chosen LOO R² and NRMSE. `metrics.json` reports final hypervolume, optional IGD+, feasible-evaluation fraction, exact-call count, evaluator time and sequential-loop time. `pareto.png` shows the first two objective coordinates; all objectives remain in the archive and metrics.

Hypervolume uses the configuration's fixed normalized reference and ideal/nadir values. A front that is empty because no point is feasible has zero hypervolume and undefined IGD+ (`null`). Benchmark IGD+ reference points must be supplied from the analytical Pareto front. For engineering comparisons, construct one common exact-verified reference union after all methods finish. Timing includes numerical model fitting and candidate search; exact-evaluator time is also reported separately.

## Numerical conventions

Each independent GP uses a Matérn-5/2 ARD kernel, standardized exact responses, bounded log length scales, amplitude and observation noise, and Cholesky inference. Numerical and semantic fits share initialization and optimizer bounds. The semantic centre is `log(0.15)+log(4)*(1-relation)` and its strength is `2*exp(-N/(5d))`. The gate compares analytic LOO negative log predictive densities exactly as in the manuscript. LOO R²/NRMSE condition on the fitted hyperparameters; they do not refit every hyperparameter for each left-out point.

Candidate scoring multiplies constraint feasibility by normalized hypervolume gain, predictive uncertainty and distance from the archive. Before discovering feasibility, aggregate lower-confidence violation rank replaces hypervolume gain. Global Sobol proposals supplement the virtual RVEA population, and already evaluated realized decisions are excluded. The selected pool and numerical-GP control expose the semantic mechanism without relabeling an approximation as a separately published baseline.
