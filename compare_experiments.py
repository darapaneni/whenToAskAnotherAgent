"""Compare the experiments: models, model classes, dependence, teams and policies.

Reads the folders written by analyse_llm_agents.py and writes, in results_llm/comparison/:

  league_table.csv        every model: class, AUC, total variation, cost per call, latency, value alone, value per dollar
  class_summary.csv       the same measures averaged within non-reasoning, general-purpose and reasoning models
  dependence.csv          mean within-outcome correlation between models of the same class, family, and across them
  teams_overview.csv      every team at every price scale: cost per loan under each policy, cheapest policy, ceiling
  headline_tests.csv      the paired differences that answer the questions of the study
  fig_frontier.png        informativeness against price, one point per model
  fig_dependence.png      correlation between models on the same application file
  fig_teams.png           cost per loan by team and policy
  fig_effort.png          informativeness and price as the reasoning effort is raised
  REPORT.md               the tables and figures on one page

    python compare_experiments.py                 # main setting
    python compare_experiments.py --referral      # the folders written with --eta 0.5 --eps 0.1
    python compare_experiments.py --mock          # fake-model outputs; NOT a result

Author: Narayana Darapaneni.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import voi_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true")
ap.add_argument("--referral", action="store_true")
ap.add_argument("--scale", type=float, default=None, help="price scale shown in fig_teams (default: the first)")
a = ap.parse_args()
HERE = Path(__file__).parent
ROOT = HERE / ("results_llm_MOCK" if a.mock else "results_llm"); SUF = "_referral" if a.referral else ""
OUT = ROOT / ("comparison" + SUF); reg, exps = C.registry(), C.experiments()
names = [n for n in exps["experiments"] if (ROOT / (n + SUF) / "agents.csv").exists()]
if not names:
    raise SystemExit(f"No analysed experiment found in {ROOT.name}/. Run analyse_llm_agents.py first.")
OUT.mkdir(parents=True, exist_ok=True)
rd = lambda n, f: pd.read_csv(ROOT / (n + SUF) / f) if (ROOT / (n + SUF) / f).exists() and (ROOT / (n + SUF) / f).stat().st_size > 2 else pd.DataFrame()

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"
CLASS_STYLE = {"non-reasoning": ("#2a78d6", "o"), "general-purpose": ("#eb6834", "s"), "reasoning": ("#1baf7a", "^"), "statistical": ("#6b6a66", "D")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight"})
def family(m):
    m = str(m)
    return "o-series" if m.startswith("o") else ("GPT-" + m[4] if m.startswith("gpt-") else "other")
short = lambda ag: ag.replace("file@", "").replace("text@", "text:").replace("policy@", "policy:")
md = []
def save(fig, name):
    if a.mock:
        fig.text(0.5, 0.5, "FAKE-MODEL OUTPUT - NOT A RESULT", ha="center", va="center", rotation=18, fontsize=20, color="#b00020", alpha=0.28, zorder=10)
    fig.savefig(OUT / name); plt.close(fig)
banner = "**FAKE-MODEL OUTPUTS. These numbers test the code and are not results.**\n\n" if a.mock else ""

# ------------------------------------------------------------------ league table
ag = pd.concat([rd(n, "agents.csv").assign(experiment=n) for n in names], ignore_index=True)
ag = ag.sort_values("n_loans", ascending=False).drop_duplicates("agent").copy()
ag["display"] = ag["model"].map(lambda m: reg["models"].get(m, {}).get("display", m)); ag["family"] = ag["model"].map(family)
for c in ("AUC", "AUC_lo", "AUC_hi", "AUC_core", "TV", "value_alone_usd", "value_per_usd"):
    if c not in ag.columns:
        ag[c] = np.nan
league = ag[ag["AUC"].notna()].sort_values("AUC", ascending=False)
cols = ["agent", "display", "class", "tier", "effort", "n_loans", "n_default", "AUC", "AUC_lo", "AUC_hi", "AUC_core", "n_core", "TV", "cost_usd", "latency_s",
        "reasoning_tokens", "unparsed", "value_alone_usd", "value_per_usd"]
league[[c for c in cols if c in league.columns]].to_csv(OUT / "league_table.csv", index=False)
file_std = league[(league["role"] == "file") & ~league["agent"].str.contains("#")]
cs = file_std.groupby("class").agg(models=("agent", "size"), loans_min=("n_loans", "min"), loans_max=("n_loans", "max"), AUC=("AUC", "mean"), best_AUC=("AUC", "max"),
                                   AUC_core=("AUC_core", "mean"), cost_usd=("cost_usd", "median"), latency_s=("latency_s", "median"),
                                   reasoning_tokens=("reasoning_tokens", "mean")).reset_index()
cs.to_csv(OUT / "class_summary.csv", index=False)

# ------------------------------------------------------------------ dependence between models (survey)
dep_rows, Rr = [], pd.DataFrame()
src = "survey" if "survey" in names else names[0]
Rr, N0 = rd(src, "corr_pooled.csv"), rd(src, "corr_n_repaid.csv")
if len(Rr):
    Rr, N0 = Rr.set_index(Rr.columns[0]), N0.set_index(N0.columns[0]); M = Rr.values; ids = list(Rr.index)
    info = {x: (reg["models"].get(C.parse_agent(x, reg)[1], {}).get("class", "statistical"), family(C.parse_agent(x, reg)[1])) for x in ids}
    allp = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids)) if not np.isnan(M[i, j])]
    isb = lambda x: x in C.BENCHMARKS
    pairs = [(i, j) for i, j in allp if not isb(ids[i]) and not isb(ids[j])]
    grp = {"same class": [(i, j) for i, j in pairs if info[ids[i]][0] == info[ids[j]][0]],
           "different class": [(i, j) for i, j in pairs if info[ids[i]][0] != info[ids[j]][0]],
           "same family": [(i, j) for i, j in pairs if info[ids[i]][1] == info[ids[j]][1]],
           "different family": [(i, j) for i, j in pairs if info[ids[i]][1] != info[ids[j]][1]],
           "language model with scorecard": [(i, j) for i, j in allp if "scorecard" in (ids[i], ids[j]) and "wordmodel" not in (ids[i], ids[j])]}
    for k, v in grp.items():
        if v:
            x = np.array([M[i, j] for i, j in v]); r = float(x.mean())
            dep_rows.append({"pairs": k, "n_pairs": len(v), "mean_correlation": r, "min": float(x.min()), "max": float(x.max()),
                             "smallest_sample_repaid": int(min(N0.values[i, j] for i, j in v)), "ceiling_independent_agents": (1 / r if r > 0 else np.inf)})
    pd.DataFrame(dep_rows).to_csv(OUT / "dependence.csv", index=False)

# ------------------------------------------------------------------ teams and tests
ps = pd.concat([rd(n, "policy_summary.csv").assign(experiment=n) for n in names], ignore_index=True)
pdiff = pd.concat([rd(n, "paired_diffs.csv").assign(experiment=n) for n in names], ignore_index=True)
tm = pd.concat([rd(n, "teams.csv").assign(experiment=n) for n in names], ignore_index=True)
over = pd.DataFrame()
if len(ps):
    over = ps.pivot_table(index=["experiment", "team", "scale"], columns="policy", values="cost_per_loan").reset_index()
    polcols = [c for c in over.columns if c not in ("experiment", "team", "scale")]
    over["cheapest_policy"] = over[polcols].idxmin(axis=1)
    over = over.merge(ps[ps.policy == "ALL"][["experiment", "team", "scale", "ceiling_ratio"]], on=["experiment", "team", "scale"], how="left")
    over = over.merge(ps[ps.policy == "DP-s"][["experiment", "team", "scale", "calls", "dearest_called"]].rename(
        columns={"calls": "calls_DP-s", "dearest_called": "dearest_called_DP-s"}), on=["experiment", "team", "scale"], how="left")
    over = over.merge(tm[["experiment", "team", "K", "r_hat", "n_eff", "n_cal", "n_test"]], on=["experiment", "team"], how="left")
    over.to_csv(OUT / "teams_overview.csv", index=False)
if len(pdiff):
    # comparisons between the prior, the scorecard, the naive product and the shrunk belief: the interval that also
    # redraws the calibration loans replaces the one over the test loans alone, which takes the likelihoods as given
    pdiff["interval"] = "test loans"
    p2 = [x.assign(experiment=n) for n, x in ((n, rd(n, "paired_diffs_cal.csv")) for n in names) if len(x)]
    if p2:
        p2 = pd.concat(p2, ignore_index=True)[["experiment", "team", "scale", "diff", "lo", "hi"]].rename(columns={"lo": "lo2", "hi": "hi2"})
        pdiff = pdiff.merge(p2, on=["experiment", "team", "scale", "diff"], how="left"); has = pdiff["lo2"].notna()
        pdiff.loc[has, "lo"] = pdiff.loc[has, "lo2"]; pdiff.loc[has, "hi"] = pdiff.loc[has, "hi2"]
        pdiff.loc[has, "interval"] = "calibration and test loans"; pdiff = pdiff.drop(columns=["lo2", "hi2"])
    pdiff["distinguishable"] = (pdiff["lo"] > 0) | (pdiff["hi"] < 0); pdiff.to_csv(OUT / "headline_tests.csv", index=False)

# ------------------------------------------------------------------ figures
def frontier():
    d = league[(league["role"].isin(["file", "scorecard"])) & ~league["agent"].str.contains("#")]
    if len(d) < 2:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(d["cost_usd"], d["AUC"], yerr=[d["AUC"] - d["AUC_lo"], d["AUC_hi"] - d["AUC"]], fmt="none", ecolor="#a8a79f", elinewidth=1.2, zorder=2)
    for cls, (col, mk) in CLASS_STYLE.items():
        x = d[d["class"] == cls]
        if len(x):
            ax.scatter(x["cost_usd"], x["AUC"], s=46, color=col, marker=mk, edgecolor="white", linewidth=1.0, label=cls, zorder=3)
    for k, (_, r) in enumerate(d.sort_values("cost_usd").iterrows()):
        ax.annotate((r["display"] if r["agent"] != "scorecard" else "scorecard") + f" (n={int(r['n_loans'])})", (r["cost_usd"], r["AUC"]),
                    xytext=[(5, 6), (5, -11), (-4, 9), (-4, -14)][k % 4], textcoords="offset points", fontsize=6, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 2)))
    ax.set_xscale("log"); ax.set_xlabel("Cost per call (USD, log scale)"); ax.set_ylabel("Discrimination (AUC), with 95% interval")
    ax.set_title("Discrimination against price: one application file, every model", loc="left", fontsize=10, color=INK)
    ax.legend(loc="lower right", fontsize=8, title="Model class", title_fontsize=8)
    save(fig, "fig_frontier.png"); return True

def dependence():
    if not len(Rr) or len(Rr) < 3:
        return False
    order = sorted(ids, key=lambda x: (list(CLASS_STYLE).index(info[x][0]), x)); Mo = pd.DataFrame(M, index=ids, columns=ids).loc[order, order].values
    n = len(order); fig, ax = plt.subplots(figsize=(0.36 * n + 2.6, 0.36 * n + 1.8))
    im = ax.imshow(np.ma.masked_invalid(Mo), cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); lab = [short(x) for x in order]
    ax.set_xticklabels(lab, rotation=90, fontsize=6.5); ax.set_yticklabels(lab, fontsize=6.5)
    cl = [info[x][0] for x in order]
    for i in range(1, n):
        if cl[i] != cl[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2); ax.axvline(i - 0.5, color="white", lw=2)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                if not np.isnan(Mo[i, j]):
                    ax.text(j, i, f"{Mo[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if Mo[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Correlation of scores within outcome class (blank: too few common loans)")
    ax.set_title("Dependence between models (ordered by class)", loc="left", fontsize=10, color=INK)
    save(fig, "fig_dependence.png"); return True

def teams_fig():
    if not len(ps):
        return False
    sc = a.scale if a.scale is not None else float(sorted(ps["scale"].unique())[0]); x = ps[ps["scale"] == sc]
    show = [("DEAREST", "Dearest agent only", "#2a78d6", "o"), ("ALL-s", "All agents", "#eb6834", "s"),
            ("CASCADE", "Cost-ordered cascade", "#1baf7a", "^"), ("DP-s", "Value-of-information policy", "#eda100", "D")]
    tlist = [t for t in x["team"].unique() if (x[x.team == t]["policy"] == "DP-s").any()]
    if not tlist:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(tlist) + 1.6))
    for pol, lab, col, mk in show:
        y = [x[(x.team == t) & (x.policy == pol)]["cost_per_loan"].mean() for t in tlist]
        ax.scatter(y, range(len(tlist)), s=44, color=col, marker=mk, edgecolor="white", linewidth=1.0, label=lab, zorder=3)
    none = x[x.policy == "NONE"]["cost_per_loan"].mean()
    ax.axvline(none, color=MUTED, lw=1, ls=(0, (4, 2))); ax.text(none, len(tlist) - 0.45, " no agent", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_yticks(range(len(tlist))); ax.set_yticklabels(tlist, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel(f"Mean realised cost per loan (USD), prices at {sc:g} times the measured token cost")
    ax.set_title("Cost per loan by team and policy", loc="left", fontsize=10, color=INK)
    fig.legend(loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig_teams.png"); return True

def effort_fig():
    order = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    d = pd.DataFrame()                                    # the experiment that varies the effort: all levels on the same loans
    for n in names:
        x = rd(n, "agents.csv")
        if len(x) and (x[x["role"] == "file"].groupby("model")["effort"].nunique() >= 3).any():
            d = x[x["role"] == "file"].copy(); break
    if not len(d):
        return False
    d["effort"] = d["effort"].fillna("")
    for c in ("AUC", "TV"):
        if c not in d.columns:
            d[c] = np.nan
    mods = [m for m, x in d.groupby("model") if x["effort"].nunique() >= 3][:3]
    if not mods:
        return False
    ycol = "TV" if d[d.model.isin(mods)]["TV"].notna().all() else "AUC"
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    present = [e for e in order if e in set(d[d.model.isin(mods)]["effort"])]
    for (m, col, mk) in zip(mods, ["#2a78d6", "#eb6834", "#1baf7a"], ["o", "s", "^"]):
        x = d[d.model == m].assign(k=lambda t: t["effort"].map(present.index)).sort_values("k")
        for ax, colname in zip(axs, [ycol, "cost_usd"]):
            ax.plot(x["k"], x[colname], color=col, marker=mk, lw=2, ms=6, mec="white", mew=1, label=reg["models"][m]["display"])
    for ax, lab in zip(axs, ["Informativeness (total variation)" if ycol == "TV" else "Discrimination (AUC)", "Cost per call (USD, log scale)"]):
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present); ax.set_xlabel("Reasoning effort"); ax.set_ylabel(lab)
    axs[1].set_yscale("log"); axs[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Asking the same model to think harder", x=0.01, ha="left", fontsize=10, color=INK); fig.tight_layout()
    save(fig, "fig_effort.png"); return True

figs = {"fig_frontier.png": frontier(), "fig_dependence.png": dependence(), "fig_teams.png": teams_fig(), "fig_effort.png": effort_fig()}

# ------------------------------------------------------------------ report
def table(d, digits):
    d = d.copy()
    for c, k in digits.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{k}f}")
    return "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values) + "\n"

md.append(f"# Comparison of the experiments{' (referral setting)' if a.referral else ''}\n\n{banner}Experiments analysed: {', '.join(names)}. "
          f"Prices in `models.json` were last checked on {reg.get('prices_checked_on')}.\n")
md.append("## 1. Which model reads a loan file best, and at what price?\n\nAUC is computed on every loan the model scored, comparing a defaulted with a repaid loan only within the same block "
          "(calibration or test), because the calibration sample is balanced and earlier in time; cheaper models were given more loans, "
          "so their intervals are narrower. `AUC_core` uses only the loans that every model scored, and compares like with like on a small sample.\n\n"
          + ("![frontier](fig_frontier.png)\n\n" if figs["fig_frontier.png"] else "")
          + table(league[[c for c in ["display", "class", "tier", "effort", "n_loans", "n_default", "AUC", "AUC_lo", "AUC_hi", "AUC_core", "TV", "cost_usd", "latency_s", "reasoning_tokens", "value_alone_usd", "agent"] if c in league.columns]],
                  {"AUC": 3, "AUC_lo": 3, "AUC_hi": 3, "AUC_core": 3, "TV": 3, "cost_usd": 5, "latency_s": 1, "reasoning_tokens": 0, "value_alone_usd": 1}))
md.append("\n## 2. Model classes\n\nFile-analyst role, standard effort. Classes differ in sample size as well as in models; read the averages with the intervals of section 1.\n\n"
          + table(cs, {"AUC": 3, "best_AUC": 3, "AUC_core": 3, "cost_usd": 5, "latency_s": 1, "reasoning_tokens": 0}))
if dep_rows:
    md.append("\n## 3. Do models make the same mistakes?\n\nMean correlation of scores within outcome class (average of defaulted and repaid loans). "
              "The last column is the ceiling 1/r of Proposition 5: the number of independent agents that any number of such agents can replace. "
              "Pairs that include a dear model rest on few loans (see `smallest_sample_repaid`).\n\n"
              + ("![dependence](fig_dependence.png)\n\n" if figs["fig_dependence.png"] else "")
              + table(pd.DataFrame(dep_rows), {"mean_correlation": 2, "min": 2, "max": 2, "ceiling_independent_agents": 1}))
if len(over):
    keep = ["experiment", "team", "K", "scale", "NONE", "SINGLE", "DEAREST", "ALL-s", "CASCADE", "MYOPIC", "DP", "DP-s", "cheapest_policy", "calls_DP-s", "dearest_called_DP-s", "ceiling_ratio", "r_hat", "n_cal", "n_test"]
    md.append("\n## 4. Teams and policies\n\nMean realised cost per loan (USD). `ceiling_ratio` is the price of all agents divided by the cost of consulting all: "
              "the largest relative saving any orchestration can deliver (Proposition 2).\n\n" + ("![teams](fig_teams.png)\n\n" if figs["fig_teams.png"] else "")
              + table(over[[c for c in keep if c in over.columns]], {**{c: 1 for c in ["NONE", "SINGLE", "DEAREST", "ALL-s", "CASCADE", "MYOPIC", "DP", "DP-s"]},
                                                                      "calls_DP-s": 2, "dearest_called_DP-s": 2, "ceiling_ratio": 4, "r_hat": 2, "scale": 0}))
if len(pdiff):
    md.append("\n## 5. Paired tests\n\nDifference in cost per loan (first policy minus second), with a 95% bootstrap interval. Column `interval` says what was "
              "redrawn: for comparisons between the prior, the scorecard alone, the naive product and the shrunk belief, the calibration loans as well as the "
              "test loans; for the others the test loans only, with the fitted likelihoods taken as given (these intervals are too narrow when the "
              "calibration sample is a few hundred loans).\n\n"
              + table(pdiff[["experiment", "team", "scale", "diff", "mean", "lo", "hi", "interval", "ceiling_usd", "distinguishable"]], {"mean": 2, "lo": 2, "hi": 2, "ceiling_usd": 3, "scale": 0}))
if figs["fig_effort.png"]:
    md.append("\n## 6. Reasoning effort\n\n![effort](fig_effort.png)\n")
sel = pd.concat([rd(n, "selection.csv").assign(experiment=n) for n in names], ignore_index=True)
if len(sel):
    md.append("\n## 7. Team selection\n\nForward selection on cross-validated cost (calibration loans); each row adds the agent that lowers the cost most.\n\n"
              + table(sel[["experiment", "scale", "step", "added", "cv_cost"]], {"cv_cost": 1, "scale": 0}))
(OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
json.dump({"experiments": names, "mock": bool(a.mock), "referral": bool(a.referral), "figures": [k for k, v in figs.items() if v]}, open(OUT / "meta.json", "w"), indent=2)
print(f"written {OUT.relative_to(HERE)}: REPORT.md, " + ", ".join(k for k, v in figs.items() if v) + ("   (FAKE-MODEL OUTPUTS - NOT A RESULT)" if a.mock else ""))
