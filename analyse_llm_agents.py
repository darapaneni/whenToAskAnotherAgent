"""Evaluate orchestration policies with language-model agents as the evidence sources.

Reads llm_outputs.csv, adds two statistical benchmarks that are not language models (a logistic scorecard on the
file fields, which can also be a team member, and a word-frequency model of the borrower's text), and for the
chosen experiment writes, in results_llm/<experiment>/:

  agents.csv          one row per agent: model, class, calls, unparsed answers, tokens, reasoning tokens, latency,
                      cost per call, AUC (see below), total variation, value alone on the median loan
  corr_*.csv          within-class correlations between agents (dependence between models on the same evidence)
  teams.csv           the teams evaluated: size, mean correlation, effective number of independent agents, prices
  policy_summary.csv  per team, price scale and policy: cost per loan, spend, calls, approvals, referrals, optimism gap
  paired_diffs.csv    paired differences between policies with bootstrap intervals; savings ceiling for All - DP
  cascade.csv         the escalation band chosen for the cost-ordered cascade
  selection.csv       the path of the team-selection algorithm
  paired_diffs_cal.csv  the differences between None, Scorecard, All and All-s again, with intervals that resample
                      the calibration loans (within outcome class) as well as the test loans
  auc_diffs.csv       paired differences in AUC between agents on the loans both scored, with bootstrap intervals
  cases.csv.gz        one row per team, price scale, policy and test loan: belief, action, calls, cost
  calibration.csv     AUC, Brier score and calibration error of the consult-all belief per team

AUC. The calibration sample is balanced and earlier in time, the test sample is random and later. The column AUC
therefore compares a defaulted with a repaid loan only within the same block (the pair-weighted mean of the AUC on
the calibration loans and on the test loans); AUC_cal and AUC_test are the two parts, and AUC_pooled, which also
compares calibration defaults with test repayers, is kept for the record only. In auc_diffs.csv the rows "all"
use the within-block AUC, "cal" and "test" the two parts, and "pooled" the pooled one.

    python analyse_llm_agents.py --experiment survey
    python analyse_llm_agents.py --experiment teams --eta 0.5 --eps 0.1      # referral setting
    python analyse_llm_agents.py --all                                       # every experiment that has data
    python analyse_llm_agents.py --experiment survey --mock                  # fake-model outputs; NOT a result

Author: Narayana Darapaneni.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import voi_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", "-e", default=None)
ap.add_argument("--all", action="store_true")
ap.add_argument("--mock", action="store_true")
ap.add_argument("--eta", type=float, default=1.0)
ap.add_argument("--eps", type=float, default=0.0)
ap.add_argument("--scales", default=None, help="multipliers of the measured cost per call (default: experiments.json)")
ap.add_argument("--bins", default="auto", help="score bins per agent: a number, or auto (5 below 500 loans per class, else 10)")
ap.add_argument("--exact-dp", action="store_true", help="robustness check: solve the dynamic programme for every loan separately (no reuse "
                "between loans with nearly equal parameters); results go to a folder ending in _exactdp")
args = ap.parse_args()

HERE = Path(__file__).parent
sys.argv = [sys.argv[0], str(args.eta), str(args.eps)]
import lc_verify as V                                    # noqa: E402
sig = V.sig
reg, exps = C.registry(), C.experiments()
names = list(exps["experiments"]) if args.all else [args.experiment]
if names == [None]:
    sys.exit("Give --experiment NAME or --all. Experiments: " + ", ".join(exps["experiments"]))
scales = [float(x) for x in (args.scales.split(",") if args.scales else exps["price_scales"])]
ROOT = HERE / ("results_llm_MOCK" if args.mock else "results_llm"); ROOT.mkdir(exist_ok=True)
suffix = ("" if args.eta >= 1.0 else "_referral") + ("_exactdp" if args.exact_dp else "")
CAL_BOOT = 10000                                         # resamples of the calibration-and-test bootstrap
AUC_BOOT, AUC_BOOT_SCORECARD = 2000, 10000               # resamples for AUC differences; more for comparisons with the scorecard
rng = np.random.default_rng(20260929)
rng_auc = np.random.default_rng(20261005)                # for the paired AUC comparisons and the test-sample intervals
CV_REPEATS = 20                                          # team selection: the cross-validated cost is averaged over this many fold splits

# ------------------------------------------------------------------ data, scorecard, outputs
df = V.load(); df["id"] = df["id"].astype(str)
tr = df["block"] == "train"
X = df[["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_util_n", "total_acc", "hist_m", "dti"]].copy()
X["log_inc"] = np.log(df["annual_inc"].clip(1000)); X["log_amt"] = np.log(df["loan_amnt"])
X["log_revol_bal"] = np.log1p(df["revol_bal"]); X = X.fillna(X[tr].median())
df["scorecard"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], df.loc[tr, "theta"]).predict_proba(X)[:, 1] * 100
df["wordmodel"] = V.fit_experiments(df)["e3"].values * 100                   # source E3 of the article: classifier fitted on the fit block;
#                                                                              its vocabulary is built from the texts of all blocks (no outcome is used for that)
BENCH = list(C.BENCHMARKS)
D = df.set_index("id")
PI_CAL = float(df.loc[df["block"] == "cal", "theta"].mean())                 # default rate of the whole calibration block
store = C.read_store(HERE / "llm_outputs_MOCK.csv" if args.mock else C.STORE, reg)
if not len(store):
    sys.exit("No outputs found. Run run_llm_agents.py first.")
test_all = df[df["block"] == "test"]
A_MED = float((test_all["L_A"] / test_all["L_R"]).median()); LR_MED = float(test_all["L_R"].median()); P0_MED = float(test_all["prior"].median())


def boot(x, n=5000):
    """Mean and 95% percentile interval of a paired bootstrap over loans (5,000 resamples, drawn in blocks)."""
    x = np.asarray(x, float); b = np.concatenate([x[rng.integers(0, len(x), (500, len(x)))].mean(axis=1) for _ in range(n // 500)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def auc(y, s):
    """Area under the ROC curve from ranks (ties get half credit); nan when one class is absent."""
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_w(y, s, c):
    """AUC that compares a defaulted with a repaid loan only within the same block: the pair-weighted mean of the AUC
    on the calibration loans (c true) and on the test loans (c false). nan when no block has both classes."""
    num = den = 0.0
    for m in (c, ~c):
        yy = y[m]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0:
            num += rankdata(s[m])[yy == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
    return float(num / den) if den else np.nan


def sampler(c, y, g):
    """Returns a function that draws one bootstrap resample of loan positions as the sample was drawn: calibration
    loans within outcome class (the calibration sample is balanced by design), test loans as one block. When a
    resample of a small test sample has no default, the test block contributes no pair to auc_w."""
    groups = [np.where(m)[0] for m in (c & (y == 1), c & (y == 0), ~c)]; groups = [x for x in groups if len(x)]
    return lambda: np.concatenate([g.choice(x, len(x)) for x in groups])


def draw_within(blk, g):
    """One bootstrap resample of loan positions, drawn within the groups of blk."""
    return np.concatenate([g.choice(np.where(blk == b)[0], int((blk == b).sum())) for b in np.unique(blk)])


def ece(p, y, nb=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)[1:-1])); b = np.digitize(p, q)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in np.unique(b)))


def realised(act, r):
    th = r["theta"]
    return {"A": th * r["L_A_real"], "R": (1 - th) * r["L_R"], "H": r["h"] + args.eps * (th * r["L_A_real"] + (1 - th) * r["L_R"])}[act]


def set_bins(th):
    n = int(min(th.sum(), (1 - th).sum()))
    V.NBINS = int(args.bins) if args.bins != "auto" else (5 if n < 500 else 10)
    return V.NBINS


def cal_weights(th):
    """Weights that restore the population class shares when the calibration sample is balanced."""
    s1 = th.mean()
    return np.where(th == 1, PI_CAL / max(s1, 1e-9), (1 - PI_CAL) / max(1 - s1, 1e-9))


def cascade(ev, bins_row, a, hh, c, z0, order, delta, zk):
    """Cost-ordered cascade: always ask the cheapest agent, then escalate while the belief is within delta
    (in log-odds) of a point where the best action changes. Beliefs use the shrunk combination."""
    sw, n, z, spend, calls = 0.0, 0, z0, 0.0, []
    for k in order:
        if n and min(abs(z - x) for x in zk) >= delta:
            break
        sw += ev.w[k][bins_row[k]]; n += 1; z = z0 + sw / (1 + (n - 1) * ev.r); spend += c[k]; calls.append(k)
    p = sig(z); return p, V.action(p, a, hh), spend, calls


def analyse(name):
    global rng, rng_auc
    rng = np.random.default_rng(20260929); rng_auc = np.random.default_rng(20261005)     # per experiment: the same numbers alone or with --all
    rng_s = np.random.default_rng(20261008); rng_c = np.random.default_rng(20261009)     # within-block AUC; calibration-and-test bootstrap
    exp = C.get_experiment(name, exps); t0 = time.time()
    OUT = ROOT / (name + suffix); OUT.mkdir(exist_ok=True)
    called = [g[0] for g in C.experiment_agents(exp, reg)]
    teams = {k: list(v) for k, v in exp.get("teams", {}).items()}
    wanted = list(dict.fromkeys(called + [a for t in teams.values() for a in t if a != "scorecard"]))
    sub = store[store["agent"].isin(wanted)]
    # an agent called by this experiment is analysed on the loans this experiment gave it, even when another
    # experiment scored more loans with the same agent: comparisons within an experiment are on matched samples
    _, _, member = C.plan(exp, reg, df)
    own = {k: set(df["id"].loc[list(v)]) for k, v in member.items()}
    sub = sub[[a not in own or l in own[a] for a, l in zip(sub["agent"], sub["loan_id"])]]
    if not len(sub):
        print(f"[{name}] no outputs yet - skipped"); return
    wide = sub.pivot_table(index="loan_id", columns="agent", values="risk_score", aggfunc="first")
    for b_ in BENCH:
        wide[b_] = D.loc[wide.index, b_]
    present = [a for a in wanted if a in wide.columns]
    if args.all and exp.get("skip_in_all"):
        OUT.rmdir() if not any(OUT.iterdir()) else None; return
    if args.all and sum(a in wide.columns for a in called) < max(1, (len(called) + 1) // 2):
        print(f"[{name}] not run yet (or run only in part) - skipped; analyse it by name to see what is there"); OUT.rmdir() if not any(OUT.iterdir()) else None; return
    print(f"\n=== {name}: {exp['title']} ===\nagents with outputs: {len(present)} of {len(wanted)}; loans: {len(wide)}")

    # ---------------------------------------------------------------- agents, one by one
    g_med = V.g_and_kinks(A_MED, args.eta * A_MED / (1 + A_MED))[0]; rows = []
    core = wide[present].dropna().index if present else wide.index           # loans scored by every model of the experiment
    for a in present + BENCH:
        s = wide[a].dropna(); d = D.loc[s.index]; cal, test = (d["block"] == "cal").values, (d["block"] == "test").values
        role, model, eff = C.parse_agent(a, reg); m = reg["models"].get(model, {})
        st = sub[sub["agent"] == a]; y = d["theta"].values
        row = {"agent": a, "role": role, "model": model or C.BENCHMARKS[a], "class": m.get("class", "statistical"), "tier": m.get("tier", ""), "effort": eff or "",
               "calls": len(st) if model else len(s), "unparsed": int(st["risk_score"].isna().sum()) if model else 0,
               "tokens_in": st["input_tokens"].mean() if model else 0, "tokens_out": st["output_tokens"].mean() if model else 0,
               "reasoning_tokens": st["reasoning_tokens"].mean() if model else 0, "latency_s": st["latency_s"].mean() if model else 0,
               "cost_usd": st["cost_usd"].mean() if model else reg.get("scorecard_cost_usd", 0.0001),
               "n_loans": len(s), "n_default": int(y.sum()), "n_repaid": int((1 - y).sum()), "n_cal": int(cal.sum()), "n_test": int(test.sum())}
        if min(row["n_default"], row["n_repaid"]) >= 5:                       # discrimination of the raw score needs no fitting: all loans are used
            bs = []
            for _ in range(2000):
                ix = rng.integers(0, len(y), len(y))
                if 0 < y[ix].sum() < len(ix):
                    bs.append(auc(y[ix], s.values[ix]))
            row.update({"AUC_pooled": roc_auc_score(y, s.values), "AUC_pooled_lo": float(np.quantile(bs, .025)), "AUC_pooled_hi": float(np.quantile(bs, .975))})
            draw = sampler(cal, y, rng_s); bw = [auc_w(y[ix], s.values[ix], cal[ix]) for ix in (draw() for _ in range(2000))]
            row.update({"AUC": auc_w(y, s.values, cal), "AUC_lo": float(np.nanquantile(bw, .025)), "AUC_hi": float(np.nanquantile(bw, .975))})
        yc = D.loc[core, "theta"].values
        if len(core) >= 20 and min(yc.sum(), (1 - yc).sum()) >= 5:
            row.update({"AUC_core": auc_w(yc, wide.loc[core, a].values, (D.loc[core, "block"] == "cal").values), "n_core": len(core)})
        th_c = d.loc[cal, "theta"].values; th_t = d.loc[test, "theta"].values
        if test.sum() >= 40 and min(th_t.sum(), (1 - th_t).sum()) >= 5:
            st_ = s[test].values; bt = [auc(th_t[ix], st_[ix]) for ix in (rng_auc.integers(0, len(th_t), len(th_t)) for _ in range(2000))]
            row.update({"AUC_test": auc(th_t, st_), "AUC_test_lo": float(np.nanquantile(bt, .025)), "AUC_test_hi": float(np.nanquantile(bt, .975))})
        if cal.sum() >= 40 and min(th_c.sum(), (1 - th_c).sum()) >= 5:
            sc_ = s[cal].values; bc = [auc(th_c[ix], sc_[ix]) for ix in (draw_within(th_c, rng_s) for _ in range(2000))]
            row.update({"AUC_cal": auc(th_c, sc_), "AUC_cal_lo": float(np.nanquantile(bc, .025)), "AUC_cal_hi": float(np.nanquantile(bc, .975))})
        if cal.sum() >= 40 and min(th_c.sum(), (1 - th_c).sum()) >= 20:
            nb = set_bins(th_c); ev1 = V.Evidence(s[cal].to_frame().reset_index(drop=True), th_c); nb = len(ev1.edges[0]) + 1   # bins actually formed
            mm = P0_MED * ev1.f1[0] + (1 - P0_MED) * ev1.f0[0]; post = P0_MED * ev1.f1[0] / mm
            voi = float((g_med(P0_MED) - (mm * g_med(post)).sum()) * LR_MED)
            row.update({"TV": ev1.TV[0], "bins": nb, "value_alone_usd": voi, "value_per_usd": voi / row["cost_usd"] if row["cost_usd"] > 0 else np.nan})
        rows.append(row)
    agents = pd.DataFrame(rows); agents.to_csv(OUT / "agents.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = [c for c in ("agent", "class", "n_loans", "n_default", "unparsed", "reasoning_tokens", "latency_s", "cost_usd", "AUC", "AUC_lo", "AUC_hi", "TV", "value_alone_usd") if c in agents.columns]
    print(agents[show].round(4).to_string())
    cost_of = agents.set_index("agent")["cost_usd"].to_dict()

    # ---------------------------------------------------------------- paired differences in discrimination
    arow = []; names_ = present + BENCH
    for i_, a1 in enumerate(names_):
        for a2 in names_[i_ + 1:]:
            both = wide[[a1, a2]].dropna(); y = D.loc[both.index, "theta"].values
            with_sc = "scorecard" in (a1, a2)            # every agent is compared with the scorecard, also on 40 loans
            if (len(both) < 100 or min(y.sum(), (1 - y).sum()) < 20) and not (with_sc and len(both) >= 40 and min(y.sum(), (1 - y).sum()) >= 5):
                continue
            blk2 = D.loc[both.index, "block"].values; c2_ = blk2 == "cal"; nbs = AUC_BOOT_SCORECARD if with_sc or len(names_) <= 8 else AUC_BOOT   # few agents: every pair gets the larger number
            for lab, mk in (("pooled", np.ones(len(y), bool)), ("test", blk2 == "test")):
                yy, s1, s2 = y[mk], both[a1].values[mk], both[a2].values[mk]
                if min(yy.sum(), (1 - yy).sum()) < 20:
                    continue
                bs = []
                for _ in range(2000):
                    ix = rng_auc.integers(0, len(yy), len(yy)); bs.append(auc(yy[ix], s1[ix]) - auc(yy[ix], s2[ix]))
                arow.append({"agent_1": a1, "agent_2": a2, "loans": lab, "n": int(mk.sum()), "n_default": int(yy.sum()), "AUC_1": auc(yy, s1), "AUC_2": auc(yy, s2),
                             "diff": auc(yy, s1) - auc(yy, s2), "lo": float(np.nanquantile(bs, .025)), "hi": float(np.nanquantile(bs, .975))})
            # the comparison used in the article: defaulted against repaid loans of the same block only
            s1, s2 = both[a1].values, both[a2].values
            draw = sampler(c2_, y, rng_s)
            bs = [auc_w(y[ix], s1[ix], c2_[ix]) - auc_w(y[ix], s2[ix], c2_[ix]) for ix in (draw() for _ in range(nbs))]
            arow.append({"agent_1": a1, "agent_2": a2, "loans": "all", "n": len(y), "n_default": int(y.sum()), "AUC_1": auc_w(y, s1, c2_), "AUC_2": auc_w(y, s2, c2_),
                         "diff": auc_w(y, s1, c2_) - auc_w(y, s2, c2_), "lo": float(np.nanquantile(bs, .025)), "hi": float(np.nanquantile(bs, .975)), "resamples": nbs})
            mk = blk2 == "cal"; yy = y[mk]
            if min(yy.sum(), (1 - yy).sum()) >= 20:
                bs = [auc(yy[ix], s1[mk][ix]) - auc(yy[ix], s2[mk][ix]) for ix in (draw_within(yy, rng_s) for _ in range(nbs))]
                arow.append({"agent_1": a1, "agent_2": a2, "loans": "cal", "n": int(mk.sum()), "n_default": int(yy.sum()), "AUC_1": auc(yy, s1[mk]), "AUC_2": auc(yy, s2[mk]),
                             "diff": auc(yy, s1[mk]) - auc(yy, s2[mk]), "lo": float(np.nanquantile(bs, .025)), "hi": float(np.nanquantile(bs, .975)), "resamples": nbs})
    pd.DataFrame(arow).to_csv(OUT / "auc_diffs.csv", index=False)

    # ---------------------------------------------------------------- dependence between agents, pair by pair
    # correlation of normal scores within outcome class, on every loan (calibration and test) that both agents scored
    cols = present + BENCH; nA = len(cols); th_all = D.loc[wide.index, "theta"].values
    Rc = {t: np.full((nA, nA), np.nan) for t in (0, 1)}; Nn = {t: np.zeros((nA, nA), int) for t in (0, 1)}
    for i_ in range(nA):
        for j_ in range(i_, nA):
            ok = wide[cols[i_]].notna().values & wide[cols[j_]].notna().values
            for t in (0, 1):
                mk = ok & (th_all == t); k = int(mk.sum()); Nn[t][i_, j_] = Nn[t][j_, i_] = k
                if k >= 8:
                    zi = V.norm.ppf(wide.loc[mk, cols[i_]].rank().values / (k + 1)); zj = V.norm.ppf(wide.loc[mk, cols[j_]].rank().values / (k + 1))
                    r = 1.0 if i_ == j_ else (float(np.corrcoef(zi, zj)[0, 1]) if zi.std() > 0 and zj.std() > 0 else np.nan)
                    Rc[t][i_, j_] = Rc[t][j_, i_] = r
    if nA >= 2:
        w0, w1 = Nn[0] * ~np.isnan(Rc[0]), Nn[1] * ~np.isnan(Rc[1])
        pooled = (np.nan_to_num(Rc[0]) * w0 + np.nan_to_num(Rc[1]) * w1) / np.where(w0 + w1 > 0, w0 + w1, np.nan)
        for nm_, M_ in (("corr_repaid", Rc[0]), ("corr_default", Rc[1]), ("corr_pooled", pooled), ("corr_n_repaid", Nn[0]), ("corr_n_default", Nn[1])):
            pd.DataFrame(M_, index=cols, columns=cols).to_csv(OUT / f"{nm_}.csv")

    # ---------------------------------------------------------------- team selection (forward, cross-validated)
    sel_rows = []
    if exp.get("select_from"):
        sf = exp["select_from"]
        pool = [f"file@{m}" for m in reg["groups"][sf[1:]]] if isinstance(sf, str) else list(sf)
        pool = [a for a in pool if a in wide.columns] + ["scorecard"]
        full = wide[pool].dropna(); dfull = D.loc[full.index]; calf = (dfull["block"] == "cal").values; thf = dfull.loc[calf, "theta"].values
    if exp.get("select_from") and len(pool) >= 3 and calf.sum() >= 100 and min(thf.sum(), (1 - thf).sum()) >= 40:
        Sc = full[calf].reset_index(drop=True); dc = dfull[calf].reset_index(drop=True)
        w = cal_weights(thf); folds = []
        rng_cv = np.random.default_rng(20261006)          # one split decides little: with a single split the selected team changed with the seed
        for _ in range(CV_REPEATS):
            fold = np.empty(len(thf), int)
            for t in (0, 1):
                idx = np.where(thf == t)[0]; fold[idx] = rng_cv.permutation(len(idx)) % 5
            folds.append(fold)
        lossA = dc["theta"].values * dc["L_A_real"].values; lossR = (1 - dc["theta"].values) * dc["L_R"].values
        lossH = dc["h"].values + args.eps * (lossA + lossR)
        z0c = np.log(dc["prior"] / (1 - dc["prior"])).values; ac = (dc["L_A"] / dc["L_R"]).values; hc = (dc["h"] / dc["L_R"]).values
        base = np.array([realised(V.action(sig(z0c[i]), ac[i], hc[i]), dc.iloc[i]) for i in range(len(dc))])

        def cv_cost(team, scale):
            out = []
            for fold in folds:
                tot = np.zeros(len(thf))
                for f in range(5):
                    a_, b_ = fold != f, fold == f
                    set_bins(thf[a_]); e = V.Evidence(Sc.loc[a_, team].reset_index(drop=True), thf[a_]); B = e.bins(Sc.loc[b_, team])
                    z = z0c[b_] + sum(e.w[k][B[:, k]] for k in range(len(team))) / (1 + (len(team) - 1) * e.r)
                    idx = np.where(b_)[0]; p = sig(z)
                    act = np.array([V.action(p[j], ac[i], hc[i]) for j, i in enumerate(idx)])
                    tot[idx] = np.where(act == "A", lossA[idx], np.where(act == "R", lossR[idx], lossH[idx]))
                out.append(float(np.average(tot, weights=w)))
            fee = scale * sum(cost_of[a] for a in team)
            return float(np.mean(out)) + fee, float(np.std(out)), np.array(out) + fee

        for scale in scales[:2]:
            team, cur = [], float(np.average(base, weights=w)); sel_rows.append({"scale": scale, "step": 0, "added": "(no agent)", "cv_cost": cur})
            cur_vec = np.full(CV_REPEATS, cur)
            while len(team) < exp.get("select_max", 4):
                cand = {a: cv_cost(team + [a], scale) for a in pool if a not in team}
                best = min(cand, key=lambda k: cand[k][0])
                if cand[best][0] >= cur:
                    break
                gain = cur_vec - cand[best][2]; second = sorted(v[0] for v in cand.values())[1] if len(cand) > 1 else np.nan
                team.append(best); cur = cand[best][0]; cur_vec = cand[best][2]
                sel_rows.append({"scale": scale, "step": len(team), "added": best, "cv_cost": cur, "sd_over_splits": cand[best][1], "splits": CV_REPEATS,
                                 "gain": float(gain.mean()), "gain_sd_over_splits": float(gain.std()), "share_splits_gain_positive": float((gain > 0).mean()),
                                 "runner_up_behind_by": float(second - cur)})
            if team:
                teams[f"selected_x{scale:g}"] = team
        pd.DataFrame(sel_rows).to_csv(OUT / "selection.csv", index=False)
        print("team selection:\n", pd.DataFrame(sel_rows).round(2).to_string())

    # ---------------------------------------------------------------- teams and policies
    trow, prow, drow, crow, kal, allcases, d2row = [], [], [], [], [], [], []
    for tname, members in teams.items():
        if any(a not in wide.columns for a in members):
            print(f"team {tname}: skipped, no outputs yet for " + ", ".join(a for a in members if a not in wide.columns)); continue
        Wt = wide[members].dropna(); d = D.loc[Wt.index]; cal, test = (d["block"] == "cal").values, (d["block"] == "test").values
        th = d.loc[cal, "theta"].values
        if cal.sum() < 40 or not 0 < th.sum() < len(th) or test.sum() < 40:
            print(f"team {tname}: skipped, too few loans ({int(cal.sum())} calibration, {int(test.sum())} test)"); continue
        nb = set_bins(th); S_cal, S_test = Wt[cal].reset_index(drop=True), Wt[test].reset_index(drop=True)
        ev = V.Evidence(S_cal, th); K = ev.K; T = d[test].reset_index(drop=True); Cc = d[cal].reset_index(drop=True)
        Bt, Bc = ev.bins(S_test), ev.bins(S_cal); base_cost = np.array([cost_of[a] for a in members], float); wc = cal_weights(th)
        best, cheap, dear = int(np.argmax(ev.TV)), int(np.argmin(base_cost)), int(np.argmax(base_cost)); order = list(np.argsort(base_cost, kind="stable"))
        sc_cal = wide.loc[Wt.index[cal], ["scorecard"]].reset_index(drop=True); sc_test = wide.loc[Wt.index[test], ["scorecard"]].reset_index(drop=True)
        ev_sc = V.Evidence(sc_cal, th); B_sc = ev_sc.bins(sc_test)[:, 0]; sc_price = reg.get("scorecard_cost_usd", 0.0001)
        trow.append({"team": tname, "agents": " + ".join(members), "K": K, "n_cal": int(cal.sum()), "n_test": int(test.sum()), "bins": nb,
                     "bins_formed": "/".join(str(len(e_) + 1) for e_ in ev.edges),
                     "r_hat": ev.r, "n_eff": K / (1 + (K - 1) * ev.r), "sum_TV": float(np.sum(ev.TV)), "price_all_usd": float(base_cost.sum()),
                     "most_informative": members[best], "cheapest": members[cheap], "dearest": members[dear]})
        cases = []
        for scale in scales:
            V._cache.clear()
            def pars(r):
                a_, hh = r["L_A"] / r["L_R"], r["h"] / r["L_R"]
                return a_, hh, base_cost * scale / r["L_R"], float(np.log(r["prior"] / (1 - r["prior"])))
            def kinks(a_, hh):
                return [float(np.log(x / (1 - x))) for x, _ in V.g_and_kinks(a_, hh)[1]]
            # escalation band of the cascade, tuned on the calibration loans
            grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99.0]; tot = np.zeros(len(grid))
            if K > 1:
                for i, r in Cc.iterrows():
                    a_, hh, c, z0 = pars(r); zk = kinks(a_, hh)
                    for j, dl in enumerate(grid):
                        p, act, spend, _ = cascade(ev, Bc[i], a_, hh, c, z0, order, dl, zk)
                        tot[j] += wc[i] * (realised(act, r) + spend * r["L_R"])
            delta = grid[int(np.argmin(tot))]; crow.append({"team": tname, "scale": scale, "delta_log_odds": delta})
            pols = [("NONE", "indep", None), ("SINGLE", "indep", best), ("ALL", "indep", None), ("ALL-s", "shrink", None)]
            if K > 1:
                pols += [("CHEAPEST", "indep", cheap), ("DEAREST", "indep", dear), ("MYOPIC", "indep", None), ("DP", "indep", None),
                         ("DP-s", "shrink", None), ("CASCADE", "shrink", None)]
            for i, r in T.iterrows():
                a_, hh, c, z0 = pars(r)
                if len(V._cache) > 1500 or args.exact_dp:
                    V._cache.clear()
                for pol, mode, fx in pols:
                    if pol == "CASCADE":
                        p, act, spend, calls = cascade(ev, Bt[i], a_, hh, c, z0, order, delta, kinks(a_, hh))
                    else:
                        base = {"SINGLE": "FIXED", "CHEAPEST": "FIXED", "DEAREST": "FIXED", "MYOPIC": "MYO", "ALL-s": "ALL", "DP-s": "DP"}.get(pol, pol)
                        p, act, spend, calls = V.run(base, ev, Bt[i], a_, hh, c, z0, mode, fixed=fx)
                    loss = realised(act, r)
                    cases.append((scale, pol, i, r["theta"], p, act, len(calls), int(dear in calls), spend * r["L_R"], loss, loss + spend * r["L_R"],
                                  min(p * r["L_A"], (1 - p) * r["L_R"], r["L_R"] * V.lH(p, a_, hh)) + spend * r["L_R"]))
                # benchmark on the same loans: the scorecard alone, with its likelihoods from the same calibration loans
                p = sig(z0 + ev_sc.w[0][B_sc[i]]); act = V.action(p, a_, hh); loss = realised(act, r); sp = sc_price * scale
                cases.append((scale, "SCORECARD", i, r["theta"], p, act, 1, 0, sp, loss, loss + sp,
                              min(p * r["L_A"], (1 - p) * r["L_R"], r["L_R"] * V.lH(p, a_, hh)) + sp))
        res = pd.DataFrame(cases, columns=["scale", "policy", "i", "theta", "p", "action", "calls", "dearest_called", "spend", "loss", "total", "model_expected"])
        for (scale, pol), x in res.groupby(["scale", "policy"], sort=False):
            gp = boot((x["total"] - x["model_expected"]).values)
            prow.append({"team": tname, "scale": scale, "policy": pol, "cost_per_loan": x["total"].mean(), "loss": x["loss"].mean(), "spend": x["spend"].mean(),
                         "calls": x["calls"].mean(), "dearest_called": x["dearest_called"].mean(), "approve": (x["action"] == "A").mean(),
                         "refer": (x["action"] == "H").mean(), "optimism_gap": gp[0], "gap_lo": gp[1], "gap_hi": gp[2],
                         "ceiling_ratio": base_cost.sum() * scale / res[(res.scale == scale) & (res.policy == "ALL")]["total"].mean()})
        for scale, x in res.groupby("scale"):
            tot = {p: y.sort_values("i")["total"].values for p, y in x.groupby("policy")}
            for a_, b_ in (("NONE", "ALL"), ("NONE", "SINGLE"), ("SINGLE", "DP"), ("ALL", "DP"), ("DP", "MYOPIC"), ("DP", "DP-s"), ("ALL", "ALL-s"),
                           ("DEAREST", "DP-s"), ("CHEAPEST", "DP-s"), ("CASCADE", "DP-s"), ("ALL-s", "DP-s"),
                           ("NONE", "ALL-s"), ("NONE", "DP-s"), ("NONE", "SCORECARD"), ("SCORECARD", "ALL-s"), ("SCORECARD", "DP-s"), ("SINGLE", "ALL-s")):
                if a_ in tot and b_ in tot:
                    m_, lo, hi = boot(tot[a_] - tot[b_])
                    drow.append({"team": tname, "scale": scale, "diff": f"{a_} - {b_}", "mean": m_, "lo": lo, "hi": hi,
                                 "ceiling_usd": base_cost.sum() * scale if (a_, b_) in (("ALL", "DP"), ("ALL-s", "DP-s")) else np.nan})
        # ---- the same static comparisons with intervals that also carry the uncertainty of the calibration sample:
        # calibration loans are redrawn within outcome class, the likelihoods and the correlation re-estimated, the
        # test loans re-decided, and the test loans redrawn
        at_, ht_ = (T["L_A"] / T["L_R"]).values, (T["h"] / T["L_R"]).values; zt_ = np.log(T["prior"] / (1 - T["prior"])).values
        lA_, lR_ = T["theta"].values * T["L_A_real"].values, (1 - T["theta"].values) * T["L_R"].values; lH_ = T["h"].values + args.eps * (lA_ + lR_)

        def loss_of(z):
            p = sig(z); act = np.stack([p * at_, 1 - p, ht_ + V.EPS * (p * at_ + 1 - p)]).argmin(0)
            return np.where(act == 0, lA_, np.where(act == 1, lR_, lH_))

        def static(e, e_sc):
            B_ = e.bins(S_test); sw = sum(e.w[k][B_[:, k]] for k in range(e.K))
            return {"NONE": loss_of(zt_), "SCORECARD": loss_of(zt_ + e_sc.w[0][e_sc.bins(sc_test)[:, 0]]),
                    "ALL": loss_of(zt_ + sw), "ALL-s": loss_of(zt_ + sw / (1 + (e.K - 1) * e.r))}

        base_l = static(ev, ev_sc); x1 = res[res.scale == scales[0]]
        for pol_ in ("NONE", "ALL", "ALL-s", "SCORECARD"):   # the vectorised losses must equal those of the loan-by-loan replay
            assert np.allclose(base_l[pol_], x1[x1.policy == pol_].sort_values("i")["loss"].values), (tname, pol_)
        pairs_ = [("NONE", "ALL-s"), ("SCORECARD", "ALL-s"), ("ALL", "ALL-s"), ("NONE", "ALL"), ("NONE", "SCORECARD"), ("SCORECARD", "ALL")]
        i1_, i0_ = np.where(th == 1)[0], np.where(th == 0)[0]; bsd = {pq: [] for pq in pairs_}; nT = len(T)
        for _ in range(CAL_BOOT):
            ix = np.concatenate([rng_c.choice(i1_, len(i1_)), rng_c.choice(i0_, len(i0_))])
            e_b = V.Evidence(S_cal.iloc[ix].reset_index(drop=True), th[ix]); esc_b = V.Evidence(sc_cal.iloc[ix].reset_index(drop=True), th[ix])
            if not np.isfinite(e_b.r):
                continue
            tb_ = static(e_b, esc_b); jt = rng_c.integers(0, nT, nT)
            for pq in pairs_:
                bsd[pq].append(float((tb_[pq[0]][jt] - tb_[pq[1]][jt]).mean()))
        fee_ = {"NONE": 0.0, "SCORECARD": sc_price, "ALL": float(base_cost.sum()), "ALL-s": float(base_cost.sum())}
        for scale in scales:
            for pq in pairs_:
                f_ = (fee_[pq[0]] - fee_[pq[1]]) * scale; v_ = np.array(bsd[pq]) + f_
                d2row.append({"team": tname, "scale": scale, "diff": f"{pq[0]} - {pq[1]}", "mean": float((base_l[pq[0]] - base_l[pq[1]]).mean()) + f_,
                              "lo": float(np.quantile(v_, .025)), "hi": float(np.quantile(v_, .975)), "boot_mean": float(v_.mean()),
                              "share_below_zero": float((v_ <= 0).mean()) if np.abs(v_).max() > 0 else np.nan, "resamples": len(v_)})
        allcases.append(res.assign(team=tname, loan_id=Wt.index[test].values[res["i"].values]))
        x0 = res[res.scale == scales[0]]
        for pol, lab in (("NONE", "prior only"), ("SINGLE", "most informative agent"), ("ALL", "all agents, naive product"), ("ALL-s", "all agents, shrinkage")):
            y = x0[x0.policy == pol]; p_, y_ = y["p"].values, y["theta"].values
            kal.append({"team": tname, "belief": lab, "AUC": roc_auc_score(y_, p_) if len(np.unique(p_)) > 1 else np.nan,
                        "brier": float(np.mean((p_ - y_) ** 2)), "ECE": ece(p_, y_), "mean_belief": float(p_.mean()), "default_rate": float(y_.mean())})
        print(f"team {tname}: K={K}, {int(cal.sum())} calibration and {int(test.sum())} test loans, r_hat={ev.r:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    for nm, rws in (("teams", trow), ("policy_summary", prow), ("paired_diffs", drow), ("paired_diffs_cal", d2row), ("cascade", crow), ("calibration", kal)):
        pd.DataFrame(rws).to_csv(OUT / f"{nm}.csv", index=False)
    if allcases:
        pd.concat(allcases)[["team", "scale", "policy", "loan_id", "theta", "p", "action", "calls", "spend", "loss", "total", "model_expected"]].round(6).to_csv(
            OUT / "cases.csv.gz", index=False)
    if prow:
        ps = pd.DataFrame(prow); piv = ps[ps.scale == scales[0]].pivot(index="team", columns="policy", values="cost_per_loan")
        print(f"cost per loan (USD) at price scale {scales[0]:g}:\n", piv.round(1).to_string())
    json.dump({"experiment": name, "title": exp["title"], "eta": args.eta, "eps": args.eps, "scales": scales, "mock": bool(args.mock),
               "loans": int(len(wide)), "agents": present, "prices_checked_on": reg.get("prices_checked_on"), "seconds": round(time.time() - t0)},
              open(OUT / "meta.json", "w"), indent=2)
    print(f"written {OUT.relative_to(HERE)}  [{time.time() - t0:.0f}s]" + ("   (FAKE-MODEL OUTPUTS - NOT A RESULT)" if args.mock else ""))


for nm in names:
    analyse(nm)
