import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm, qmc, rankdata

from .gp import GaussianProcess


def prepare(args):
    config = json.loads(Path(args.config).read_text())
    variables = config["variables"]
    responses = config["objectives"] + config.get("constraints", [])
    if args.blank:
        relation = np.zeros((len(responses), len(variables)))
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.encoder)
        x = encoder.encode(variables, normalize_embeddings=True)
        y = encoder.encode(responses, normalize_embeddings=True)
        if args.shuffled:
            x = np.random.default_rng(args.seed).permutation(x)
        relation = np.clip((1 + y @ x.T) / 2, 0, 1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "semantics.npz", relation=relation, blank=args.blank)
    (out / "problem.json").write_text(json.dumps(config, indent=2))


def fit_models(x, response, relation, blank, no_gate, no_decay):
    models, gates, diagnostics = [], [], []
    strength = 2 if no_decay else 2 * np.exp(-len(x) / (5 * x.shape[1]))
    for q in range(response.shape[1]):
        numerical = GaussianProcess().fit(x, response[:, q])
        if blank:
            chosen, gate, semantic_score = numerical, False, None
        else:
            prior = np.log(0.15) + np.log(4) * (1 - relation[q])
            semantic = GaussianProcess(prior, strength).fit(x, response[:, q])
            gate = bool(no_gate or semantic.loo_nlpd <= 1.05 * numerical.loo_nlpd)
            chosen = semantic if gate else numerical
            semantic_score = semantic.loo_nlpd
        models.append(chosen)
        gates.append(gate)
        diagnostics.append({"numerical_loo_nlpd": numerical.loo_nlpd, "semantic_loo_nlpd": semantic_score,
                            "chosen_loo_r2": chosen.loo_r2, "chosen_loo_nrmse": chosen.loo_nrmse})
    return models, gates, diagnostics


def scale01(values):
    return (values - values.min()) / max(float(np.ptp(values)), 1e-12)


def nondominated(objectives, constraints):
    feasible = np.all(constraints <= 0, axis=1) if constraints.shape[1] else np.ones(len(objectives), bool)
    indices = np.flatnonzero(feasible)
    if not len(indices):
        return indices
    return indices[NonDominatedSorting().do(objectives[indices], only_non_dominated_front=True)]


def run(args):
    root = Path(args.data)
    config = json.loads((root / "problem.json").read_text())
    semantics = np.load(root / "semantics.npz", allow_pickle=False)
    dimensions, objectives = len(config["variables"]), len(config["objectives"])
    custom = config.get("evaluator")
    if custom:
        module, function = custom.split(":")
        exact = getattr(importlib.import_module(module), function)
        lower, upper = np.array(config["lower"]), np.array(config["upper"])
    else:
        problem = get_problem(config["problem"], n_var=dimensions, **({"n_obj": objectives} if config["problem"].startswith(("dtlz", "wfg")) else {}))
        lower, upper = problem.xl, problem.xu

        def exact(design):
            values = problem.evaluate(design[None], return_as_dictionary=True)
            return values["F"][0], values.get("G", np.empty((1, 0)))[0]

    reference = np.array(config["reference"])
    ideal, nadir = np.array(config["ideal"]), np.array(config["nadir"])
    metric_scale = np.maximum(nadir - ideal, 1e-8)
    hypervolume = HV(ref_point=reference)
    rng = np.random.default_rng(args.seed)
    designs = [qmc.LatinHypercube(dimensions, seed=args.seed + i).random(5 * dimensions) for i in range(10)]
    x = max(designs, key=lambda points: pdist(points).min())
    if args.budget < len(x):
        raise ValueError("Exact budget must cover the 5d initial design")
    y, constraints, integer = [], [], config.get("integer_indices", [])

    def evaluate(point):
        physical = lower + point * (upper - lower)
        if integer:
            physical[integer] = np.rint(physical[integer])
        realized = (physical - lower) / (upper - lower)
        f, g = exact(physical)
        f, g = np.asarray(f).reshape(-1), np.asarray(g).reshape(-1)
        if len(f) != objectives or len(g) != len(config.get("constraints", [])) or not np.isfinite(np.r_[f, g]).all():
            raise ValueError("Exact evaluator returned invalid objective/constraint values")
        return realized, f, g

    realized = []
    exact_seconds = 0
    for point in x:
        begin = perf_counter()
        point, f, g = evaluate(point)
        exact_seconds += perf_counter() - begin
        realized.append(point)
        y.append(f)
        constraints.append(g)
    x, y, constraints = np.array(realized), np.array(y), np.array(constraints)
    directions = get_reference_directions("energy", objectives, args.population, seed=args.seed)
    history = []
    begin = perf_counter()
    while len(x) < args.budget:
        models, gates, diagnostics = fit_models(x, np.concatenate([y, constraints], 1), semantics["relation"], bool(semantics["blank"]), args.no_gate, args.no_decay)

        class VirtualProblem(Problem):
            def __init__(self):
                super().__init__(n_var=dimensions, n_obj=objectives, xl=0, xu=1)

            def _evaluate(self, points, out, *unused, **kwargs):
                out["F"] = np.column_stack([model.predict(points)[0] for model in models[:objectives]])

        candidates = minimize(VirtualProblem(), RVEA(ref_dirs=directions), ("n_gen", args.generations), seed=int(rng.integers(2 ** 31)), verbose=False).pop.get("X")
        global_points = qmc.Sobol(dimensions, scramble=True, seed=int(rng.integers(2 ** 31))).random_base2(6)
        candidates = np.unique(np.r_[candidates, global_points], axis=0)
        if integer:
            physical = lower + candidates * (upper - lower)
            physical[:, integer] = np.rint(physical[:, integer])
            candidates = np.unique((physical - lower) / (upper - lower), axis=0)
        distance = cdist(candidates, x).min(1)
        candidates, distance = candidates[distance > 1e-9], distance[distance > 1e-9]
        if not len(candidates):
            raise ValueError("No unevaluated design remains in the candidate pool")
        prediction = [model.predict(candidates) for model in models]
        mean = np.column_stack([p[0] for p in prediction])
        std = np.column_stack([p[1] for p in prediction])
        feasibility = norm.cdf(-mean[:, objectives:] / np.maximum(std[:, objectives:], 1e-9)).prod(1)
        front = nondominated(y, constraints)
        archive = (y[front] - ideal) / metric_scale
        if len(front):
            current_hv = hypervolume(archive)
            gain = np.array([max(0, hypervolume(np.r_[archive, ((p - ideal) / metric_scale)[None]]) - current_hv) for p in mean[:, :objectives]])
        else:
            violation = np.maximum(mean[:, objectives:] - std[:, objectives:], 0).sum(1)
            gain = -rankdata(violation)
        robust_scale = np.maximum(np.quantile(y, 0.75, axis=0) - np.quantile(y, 0.25, axis=0), 1e-8)
        uncertainty = (std[:, :objectives] / robust_scale).mean(1)
        eta = 0.25 + 0.65 * len(x) / args.budget
        acquisition = feasibility * (eta * scale01(gain) + (1 - eta) * scale01(uncertainty) + 0.1 * scale01(distance))
        point = candidates[int(acquisition.argmax())]
        exact_start = perf_counter()
        point, f, g = evaluate(point)
        exact_seconds += perf_counter() - exact_start
        x, y, constraints = np.r_[x, point[None]], np.r_[y, f[None]], np.r_[constraints, g[None]]
        front = nondominated(y, constraints)
        history.append({"evaluations": len(x), "hypervolume": float(hypervolume((y[front] - ideal) / metric_scale)) if len(front) else 0,
                        "semantic_gate": gates, "loo": diagnostics})
    elapsed = perf_counter() - begin
    front = nondominated(y, constraints)
    feasible = np.all(constraints <= 0, 1)
    metrics = {"hypervolume": float(hypervolume((y[front] - ideal) / metric_scale)) if len(front) else 0,
               "feasible_evaluation_fraction": float(feasible.mean()), "exact_evaluations": len(x),
               "exact_seconds": exact_seconds, "sequential_loop_seconds": elapsed}
    if args.front:
        reference_front = np.loadtxt(args.front, delimiter=",")
        metrics["igd_plus"] = float(IGDPlus((reference_front - ideal) / metric_scale)((y[front] - ideal) / metric_scale)) if len(front) else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "archive.npz", x=x, objectives=y, constraints=constraints, pareto_indices=front)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y[front, 0], y[front, 1], s=18)
    ax.set(xlabel="Objective 1", ylabel="Objective 2")
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", default="configs/zdt1.json")
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--blank", action="store_true")
    p.add_argument("--shuffled", action="store_true")
    p.add_argument("--seed", type=int, default=2026001)
    p.set_defaults(function=prepare)
    p = sub.add_parser("run")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026001)
    p.add_argument("--front")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--no-decay", action="store_true")
    p.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)
