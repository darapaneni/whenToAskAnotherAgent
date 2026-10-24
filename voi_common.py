"""Shared pieces of the language-model experiments: registry, experiment definitions, agents, samples, costs.

Author: Narayana Darapaneni.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
STORE = HERE / "llm_outputs.csv"
STORE_COLS = ["loan_id", "block", "agent", "role", "model", "effort", "risk_score", "input_tokens", "output_tokens",
              "reasoning_tokens", "latency_s", "status", "raw", "model_returned", "time_utc"]
ROLES = ("text", "file", "policy")
# statistical benchmarks, not language models: a logistic scorecard on the file fields, and a logistic regression
# on the word frequencies of what the borrower wrote (source E3 of the article)
BENCHMARKS = {"scorecard": "logistic regression on file fields", "wordmodel": "logistic regression on word frequencies"}
SAMPLE_SEED = 20261004


def load_env(path=HERE / ".env"):
    """Read KEY=value lines from .env into the environment (existing variables are not overwritten)."""
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1); k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.lower().startswith("export "):
            k = k[7:].strip()
        if v and not os.environ.get(k):
            os.environ[k] = v
    return True


def registry(path=None):
    return json.loads(Path(path or HERE / "models.json").read_text(encoding="utf-8"))


def experiments(path=None):
    return json.loads(Path(path or HERE / "experiments.json").read_text(encoding="utf-8"))


def get_experiment(name, exps=None):
    exps = exps or experiments()
    if name not in exps["experiments"]:
        sys.exit(f"Unknown experiment '{name}'. Available: {', '.join(exps['experiments'])}")
    return exps["experiments"][name]


# ------------------------------------------------------------------ agents
def agent_key(role, model, effort, reg):
    """role@model, with #effort appended only when the effort differs from the registry's standard effort."""
    std = reg["models"][model].get("effort")
    return f"{role}@{model}" + ("" if effort in (None, std) else f"#{effort}")


def parse_agent(key, reg):
    """Return (role, model, effort) of an agent key; a statistical benchmark is (its name, None, None)."""
    if key in BENCHMARKS:
        return key, None, None
    role, rest = key.split("@", 1)
    model, _, eff = rest.partition("#")
    if role not in ROLES or model not in reg["models"]:
        raise ValueError(f"bad agent '{key}': role must be one of {ROLES} and the model must be in models.json")
    return role, model, (eff or reg["models"][model].get("effort"))


def expand_models(spec, reg):
    if isinstance(spec, str) and spec.startswith("@"):
        return list(reg["groups"][spec[1:]])
    return list(spec)


def experiment_agents(exp, reg):
    """Ordered list of (agent key, role, model, effort, n_cal, n_test) that an experiment calls. An agent named in
    two blocks gets the larger sample."""
    out = {}
    for block in exp["calls"]:
        for model in expand_models(block["models"], reg):
            for role in block["roles"]:
                for eff in block.get("efforts", [reg["models"][model].get("effort")]):
                    key = agent_key(role, model, eff, reg); old = out.get(key)
                    nc, nt = block.get("n_cal", 0), block.get("n_test", 0)
                    out[key] = (key, role, model, eff, max(nc, old[4]) if old else nc, max(nt, old[5]) if old else nt)
    return list(out.values())


def agent_blocks(exp, reg):
    """Position (0, 1, ...) of the first block under 'calls' in which each agent appears. Calls are made block by
    block in this order, so that if the money runs out it is the last blocks that lose loans."""
    out = {}
    for b, block in enumerate(exp["calls"]):
        for model in expand_models(block["models"], reg):
            for role in block["roles"]:
                for eff in block.get("efforts", [reg["models"][model].get("effort")]):
                    out.setdefault(agent_key(role, model, eff, reg), b)
    return out


# ------------------------------------------------------------------ samples
def sample_ids(df, n_cal, n_test):
    """Nested samples. Calibration: n_cal loans, half defaulted and half repaid (0 = the whole block, unbalanced).
    Test: the first n_test loans of a fixed random order (0 = the whole block). A larger sample contains every
    smaller one. Returns loan indices in processing order, calibration and test interleaved."""
    rng = np.random.default_rng(SAMPLE_SEED)
    cal1 = rng.permutation(df.index[(df["block"] == "cal") & (df["theta"] == 1)].values)
    cal0 = rng.permutation(df.index[(df["block"] == "cal") & (df["theta"] == 0)].values)
    test = rng.permutation(df.index[df["block"] == "test"].values)
    if n_cal:
        h = n_cal // 2; c1, c0 = cal1[:min(h, len(cal1))], cal0[:min(n_cal - h, len(cal0))]
    else:
        c1, c0 = cal1, cal0
    t = test[:n_test] if n_test else test
    cal = np.empty(len(c1) + len(c0), dtype=int); k = min(len(c1), len(c0))
    cal[0:2 * k:2], cal[1:2 * k:2] = c1[:k], c0[:k]; cal[2 * k:] = np.concatenate([c1[k:], c0[k:]])
    order, i, j = [], 0, 0
    while i < len(cal) or j < len(t):                    # interleave in proportion, so a partial run stays usable
        if j < len(t) and (i >= len(cal) or j * max(len(cal), 1) <= i * max(len(t), 1)):
            order.append(t[j]); j += 1
        else:
            order.append(cal[i]); i += 1
    return order


def plan(exp, reg, df):
    """Processing order of the loans of an experiment and, for every agent, the set of loans it is given."""
    agents = experiment_agents(exp, reg)
    order = sample_ids(df, max(g[4] for g in agents), max(g[5] for g in agents))
    cache, member = {}, {}
    for g in agents:
        if (g[4], g[5]) not in cache:
            cache[(g[4], g[5])] = set(sample_ids(df, g[4], g[5]))
        member[g[0]] = cache[(g[4], g[5])]
    return agents, order, member


# ------------------------------------------------------------------ costs and the output store
def call_cost(model, n_in, n_out, reg):
    """USD for one call from token counts; reasoning tokens are part of the output tokens."""
    m = reg["models"].get(model)
    if m is None or m.get("price_in") is None:
        return np.nan
    return (n_in * m["price_in"] + n_out * m["price_out"]) / 1e6


def read_store(path=STORE, reg=None):
    if not Path(path).exists():
        return pd.DataFrame(columns=STORE_COLS + ["cost_usd"])
    d = pd.read_csv(path, dtype={"loan_id": str, "effort": str, "status": str, "raw": str})
    reg = reg or registry()
    pin = d["model"].map(lambda m: (reg["models"].get(m) or {}).get("price_in", np.nan))
    pout = d["model"].map(lambda m: (reg["models"].get(m) or {}).get("price_out", np.nan))
    d["cost_usd"] = (d["input_tokens"] * pin + d["output_tokens"] * pout) / 1e6
    return d


# ------------------------------------------------------------------ spending
LEDGER = HERE / "spend_ledger.csv"                       # test calls of check_setup.py, which are not in the output store


def ledger_total(path=None):
    path = Path(path or LEDGER)
    if not path.exists():
        return 0.0
    return float(pd.read_csv(path)["usd"].sum())


def ledger_add(what, usd, path=None):
    path = Path(path or LEDGER); new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("time_utc,what,usd\n")
        f.write(f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')},{what},{usd:.6f}\n")


def total_spent(store, ledger=True):
    """USD spent so far: every logged call at the prices of models.json, plus the ledger (setup test calls and
    attempts that ended without an answer, the latter counted at their worst case)."""
    return float(np.nansum(store["cost_usd"].values)) + (ledger_total() if ledger else 0.0)


def unpriced(store):
    """Models with logged calls that have no price in models.json; their calls are missing from the total."""
    return sorted(set(store.loc[store["cost_usd"].isna(), "model"].astype(str))) if len(store) else []


def experiment_spent(store, agents, member, df):
    """USD of the calls in the store that belong to an experiment (calls shared with another experiment count here too)."""
    if not len(store):
        return 0.0
    ids = df["id"]; tot = 0.0
    for g in agents:
        loans = set(ids.loc[list(member[g[0]])]); x = store[store["agent"] == g[0]]
        tot += float(np.nansum(x.loc[x["loan_id"].isin(loans), "cost_usd"].values))
    return tot


def output_budget(model, eff, reg):
    """Largest number of output tokens (answer plus reasoning) a call may use; sent to the provider as a hard limit."""
    m = reg["models"][model]; budget = m.get("max_output_tokens", 120)
    if eff is not None and eff != m.get("effort"):
        budget = reg.get("effort_output_budget", {}).get(eff, budget)
    return budget


INPUT_MARGIN = 1.5                                       # input tokens are estimated as characters / 4; this covers the error


def worst_case_cost(model, eff, reg, n_in):
    """Upper bound on the cost of one call: generous input plus the whole output budget."""
    m = reg["models"][model]
    return (INPUT_MARGIN * n_in * m["price_in"] + output_budget(model, eff, reg) * m["price_out"]) / 1e6
