"""Sample, evidence sources, losses and policies on Lending Club 2007-2011 (LoanStats3a).

Experiments are four statistical models built on disjoint information:
  e1 credit history, e2 repayment capacity, e3 borrower text (TF-IDF), e4 the lender's sub-grade.
They stand in for agents; no LLM is called. Costs per call are set by a grid.

Outputs (results_eta*_eps*/): the sample, the sources and the policy results of Sections 4.1-4.4 of the article.

Author: Narayana Darapaneni.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
RNG = np.random.default_rng(20260929)
PHI, NBINS, ALPHA = 0.5, 10, 0.5
ETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUT = ROOT / f"results_eta{ETA}_eps{EPS}"; OUT.mkdir(exist_ok=True)
ZG = np.linspace(-10, 10, 401)
sig = lambda z: 1 / (1 + np.exp(-z))

# ------------------------------------------------------------------ data
def load():
    df = pd.read_csv(ROOT / "data" / "LoanStats3a.csv", low_memory=False)
    df = df[df["id"].astype(str).str.isdigit()].copy()
    df = df[df["term"].str.contains("36")]
    st = df["loan_status"].str.replace("Does not meet the credit policy.  Status:", "", regex=False)
    df = df[st.isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["theta"] = (~st.loc[df.index].eq("Fully Paid")).astype(int)
    df["date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["block"] = np.select([df["date"] <= "2010-06-30", df["date"] <= "2011-03-31"], ["train", "cal"], "test")
    num = lambda c: pd.to_numeric(df[c].astype(str).str.rstrip("%"), errors="coerce")
    df["revol_util_n"] = num("revol_util"); df["int_rate_n"] = num("int_rate")
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["hist_m"] = (df["date"] - ecl).dt.days / 30.44
    df["emp_y"] = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float)[0]
    df["text"] = (df["title"].fillna("") + " " + df["desc"].fillna("").map(
        lambda t: re.sub(r"<[^>]+>|Borrower added on \S+ >", " ", str(t)))).str.lower()
    g = df["sub_grade"].astype(str)
    df["grade_ord"] = (g.str[0].map({c: i for i, c in enumerate("ABCDEFG")}) * 5 + g.str[1].astype(int)).astype(float)
    # losses (Section 2.1): decision-time quantities from the training block only
    rec = df["total_rec_prncp"] + df["recoveries"] - df["collection_recovery_fee"]
    df["L_A_real"] = np.clip(df["funded_amnt"] - rec, 0, None)
    tr = df[df["block"] == "train"]
    lgd = (tr["L_A_real"] / tr["funded_amnt"])[tr["theta"] == 1].groupby(tr["purpose"]).mean()
    df["lgd_hat"] = df["purpose"].map(lgd).fillna(lgd.mean())
    df["L_A"] = df["lgd_hat"] * df["funded_amnt"]
    df["L_R"] = (1 - PHI) * np.clip(df["installment"] * 36 - df["funded_amnt"], 1, None)
    df["h_bar"] = df["L_A"] * df["L_R"] / (df["L_A"] + df["L_R"])
    df["h"] = ETA * df["h_bar"]
    ca = df[df["block"] == "cal"]          # most recent block before the test period
    pri = ca.groupby("purpose")["theta"].mean()
    df["prior"] = df["purpose"].map(pri).fillna(ca["theta"].mean()).clip(0.01, 0.99)
    return df.reset_index(drop=True)

# ------------------------------------------------------------- experiments
def fit_experiments(df):
    tr = df["block"] == "train"
    y = df.loc[tr, "theta"].values
    out = {}
    X1 = df[["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_util_n", "total_acc", "hist_m"]].copy()
    X1["log_revol_bal"] = np.log1p(df["revol_bal"]); X1["no_delinq_record"] = df["mths_since_last_delinq"].isna().astype(float)
    X2 = pd.concat([pd.DataFrame({"log_inc": np.log(df["annual_inc"].clip(1000)), "dti": df["dti"], "emp_y": df["emp_y"],
                                  "log_amt": np.log(df["loan_amnt"]), "amt_inc": df["loan_amnt"] / df["annual_inc"].clip(1000)}),
                    pd.get_dummies(df[["home_ownership", "verification_status", "purpose"]].astype(str), dtype=float)], axis=1)
    for name, X in (("e1", X1), ("e2", X2), ("e4", df[["grade_ord"]])):
        X = X.fillna(X[tr].median())
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], y)
        out[name] = m.predict_proba(X)[:, 1]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
    T = vec.fit_transform(df["text"])
    m = LogisticRegression(max_iter=3000, C=0.5).fit(T[tr.values], y)
    out["e3"] = m.predict_proba(T)[:, 1]
    S = pd.DataFrame(out)[["e1", "e2", "e3", "e4"]]
    return S

# ------------------------------------------------------------ evidence model
class Evidence:
    def __init__(self, S_cal, th):
        self.K = S_cal.shape[1]; self.names = list(S_cal.columns)
        n1, n0 = th.sum(), (1 - th).sum()
        self.edges, self.f1, self.f0, self.w, self.TV, self.D1, self.D0 = [], [], [], [], [], [], []
        for k, c in enumerate(self.names):
            x = S_cal[c].values
            e = np.unique(np.quantile(x, np.linspace(0, 1, NBINS + 1)[1:-1])); B = len(e) + 1
            b = np.digitize(x, e)
            f1 = (np.bincount(b[th == 1], minlength=B) + ALPHA) / (n1 + B * ALPHA)
            f0 = (np.bincount(b[th == 0], minlength=B) + ALPHA) / (n0 + B * ALPHA)
            self.edges.append(e); self.f1.append(f1); self.f0.append(f0); self.w.append(np.log(f1 / f0))
            self.TV.append(0.5 * np.abs(f1 - f0).sum())
            self.D1.append(float(np.sum(f1 * np.log(f1 / f0)))); self.D0.append(float(np.sum(f0 * np.log(f0 / f1))))
        R = []
        for t in (0, 1):
            X = S_cal[th == t]; Z = norm.ppf(X.rank().values / (len(X) + 1)); R.append(np.atleast_2d(np.corrcoef(Z, rowvar=False)))
        self.R = R
        off = np.concatenate([Rt[np.triu_indices(self.K, 1)] for Rt in R])
        self.r = float(np.clip(off.mean(), 0, 0.95)) if off.size else 0.0      # a single agent has no correlation
    def bins(self, S):
        return np.column_stack([np.digitize(S.iloc[:, k].values, self.edges[k]) for k in range(self.K)])

# ---------------------------------------------------------------- losses
def lH(p, a, hh): return hh + EPS * (p * a + (1 - p))
def g_and_kinks(a, hh):
    """normalised by L_R: lA=p a, lR=1-p, lH=hh+EPS(pa+1-p). Returns g and its kinks (location, slope drop)."""
    def g(p): return np.minimum(np.minimum(p * a, 1 - p), lH(p, a, hh))
    pstar = 1 / (1 + a)
    if lH(pstar, a, hh) < a / (1 + a):   # referral active (Proposition 3)
        sH = EPS * (a - 1)                            # slope of lH in p
        c1 = (hh + EPS) / (a - sH)                    # p a = hh + EPS + sH p
        c2 = (1 - hh - EPS) / (1 + sH)                # 1 - p = hh + EPS + sH p
        kinks = [(c1, a - sH), (c2, sH + 1.0)]
    else:
        kinks = [(pstar, a + 1.0)]
    return g, kinks

def action(p, a, hh):
    l = np.array([p * a, 1 - p, lH(p, a, hh)]); return "ARH"[int(np.argmin(l))]

# ------------------------------------------------------------------ dynamic programme (Bellman recursion of Section 2.2)
_cache = {}
def solve(ev, a, hh, c, z0, rho_mode):
    key = (round(a, 2), round(hh, 3), tuple(np.round(c, 5)), round(z0, 2), rho_mode)
    if key in _cache: return _cache[key]
    K = ev.K; p = sig(ZG); g, _ = g_and_kinks(a, hh); G = g(p)
    rho = (lambda n: 1.0) if rho_mode == "indep" else (lambda n: 1.0 / (1 + (n - 1) * ev.r) if n > 0 else 1.0)
    V = {(1 << K) - 1: G}; Q = {}
    for mask in sorted(range(1 << K), key=lambda m: -bin(m).count("1")):
        if mask == (1 << K) - 1: Q[mask] = {}; continue
        n = bin(mask).count("1"); sw = (ZG - z0) / rho(n) if n else np.zeros_like(ZG)
        best = G.copy(); Q[mask] = {}
        for k in range(K):
            if mask >> k & 1: continue
            f1, f0, w = ev.f1[k], ev.f0[k], ev.w[k]
            m = p[:, None] * f1 + (1 - p[:, None]) * f0
            zn = z0 + rho(n + 1) * (sw[:, None] + w)
            vn = np.interp(np.clip(zn, ZG[0], ZG[-1]), ZG, V[mask | (1 << k)])
            q = c[k] + (m * vn).sum(1); Q[mask][k] = q; best = np.minimum(best, q)
        V[mask] = best
    _cache[key] = (V, Q, G); return _cache[key]

def run(policy, ev, bins_row, a, hh, c, z0, rho_mode="indep", fixed=None):
    K = ev.K; mask, sw, n, z, spend, calls = 0, 0.0, 0, z0, 0.0, []
    rho = (lambda n: 1.0) if rho_mode == "indep" else (lambda n: 1.0 / (1 + (n - 1) * ev.r) if n > 0 else 1.0)
    if policy in ("DP", "MYO"): V, Q, G = solve(ev, a, hh, c, z0, rho_mode)
    while n < K:
        if policy == "ALL": k = n
        elif policy == "NONE": k = None
        elif policy == "FIXED": k = fixed if n == 0 else None
        elif policy == "DP":
            gz = np.interp(z, ZG, G); k, best = None, gz - 1e-12
            for kk, q in Q[mask].items():
                qz = np.interp(z, ZG, q)
                if qz < best: k, best = kk, qz
        elif policy == "MYO":
            p = sig(z); g, _ = g_and_kinks(a, hh); k, gain = None, 0.0
            for kk in range(K):
                if mask >> kk & 1: continue
                m = p * ev.f1[kk] + (1 - p) * ev.f0[kk]
                pn = sig(z0 + rho(n + 1) * (sw + ev.w[kk]))
                v = g(p) - (m * g(pn)).sum() - c[kk]
                if v > gain: k, gain = kk, v
        if k is None: break
        sw += ev.w[k][bins_row[k]]; n += 1; mask |= 1 << k; z = z0 + rho(n) * sw; spend += c[k]; calls.append(k)
    p = sig(z); return p, action(p, a, hh), spend, calls

# -------------------------------------------------------------- VOI checks (Theorem 1)
def _ratio_at(q, f1, f0, g, a, tv):
    """VOI / ((a+1) q(1-q) TV) evaluated exactly at the belief q (no grid interpolation)."""
    m = q * f1 + (1 - q) * f0
    return (g(q) - (m * g(q * f1 / m)).sum()) / ((a + 1) * q * (1 - q) * tv)

def voi_checks(ev, a, hh):
    p = np.linspace(0.001, 0.999, 999); g, kinks = g_and_kinks(a, hh); rows = []
    for k in range(ev.K):
        f1, f0 = ev.f1[k], ev.f0[k]
        m = p[:, None] * f1 + (1 - p[:, None]) * f0
        post = p[:, None] * f1 / m
        voi = g(p) - (m * g(post)).sum(1)
        ident = sum(D / 2 * ((m * np.abs(post - cj)).sum(1) - np.abs(p - cj)) for cj, D in kinks)
        bound = (a + 1) * p * (1 - p) * ev.TV[k]
        rows.append({"experiment": ev.names[k], "TV": ev.TV[k], "max_identity_error": float(np.max(np.abs(voi - ident))),
                     "max_ratio_voi_to_bound": float(np.max(voi / np.maximum(bound, 1e-15))),
                     "ratio_at_pstar": float(_ratio_at(1 / (1 + a), f1, f0, g, a, ev.TV[k])),
                     "min_voi": float(voi.min())})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    df = load(); S = fit_experiments(df)
    cal, test = df["block"] == "cal", df["block"] == "test"
    ev = Evidence(S[cal].reset_index(drop=True), df.loc[cal, "theta"].values)
    samp = df.groupby("block").agg(n=("theta", "size"), defaults=("theta", "sum"), rate=("theta", "mean"),
                                   mean_funded=("funded_amnt", "mean"), mean_LA=("L_A", "mean"), mean_LR=("L_R", "mean"))
    samp.reindex(["train", "cal", "test"]).to_csv(OUT / "sample.csv")
    exp = pd.DataFrame({"experiment": ev.names, "AUC_test": [roc_auc_score(df.loc[test, "theta"], S.loc[test, c]) for c in ev.names],
                        "TV": ev.TV, "D1": ev.D1, "D0": ev.D0})
    exp.to_csv(OUT / "experiments.csv", index=False)
    pd.DataFrame(ev.R[1], index=ev.names, columns=ev.names).to_csv(OUT / "R_default.csv")
    pd.DataFrame(ev.R[0], index=ev.names, columns=ev.names).to_csv(OUT / "R_repaid.csv")
    a_med = float((df.loc[test, "L_A"] / df.loc[test, "L_R"]).median()); hh_med = ETA * a_med / (1 + a_med)
    vc = voi_checks(ev, a_med, hh_med); vc.to_csv(OUT / "voi_checks.csv", index=False)
    vc2 = voi_checks(ev, a_med, 10.0); vc2.to_csv(OUT / "voi_checks_nodefer.csv", index=False)

    T = df[test].reset_index(drop=True); Bt = ev.bins(S[test].reset_index(drop=True))
    if NSAMP:
        idx = np.sort(RNG.choice(len(T), NSAMP, replace=False)); T = T.loc[idx].reset_index(drop=True); Bt = Bt[idx]
    best_single = int(np.argmax(ev.TV))
    rows, cert = [], []
    for cost in (0.01, 1.0, 5.0, 20.0, 50.0):
        _cache.clear()
        for i, r in T.iterrows():
            a, hh = r["L_A"] / r["L_R"], r["h"] / r["L_R"]; c = np.full(ev.K, cost / r["L_R"]); z0 = np.log(r["prior"] / (1 - r["prior"]))
            for pol, mode in (("NONE", "indep"), ("FIXED", "indep"), ("ALL", "indep"), ("DP", "indep"), ("MYO", "indep"), ("DP", "shrink"), ("ALL", "shrink")):
                p, act, spend, calls = run(pol, ev, Bt[i], a, hh, c, z0, mode, fixed=best_single)
                th = r["theta"]
                loss = {"A": th * r["L_A_real"], "R": (1 - th) * r["L_R"],
                        "H": r["h"] + EPS * (th * r["L_A_real"] + (1 - th) * r["L_R"])}[act]
                lA, lR = p * r["L_A"], (1 - p) * r["L_R"]
                rows.append({"cost": cost, "policy": pol + ("" if mode == "indep" else "-shrink"), "theta": th, "p": p,
                             "prior": r["prior"], "action": act, "calls": len(calls), "spend": spend * r["L_R"],
                             "loss": loss, "total": loss + spend * r["L_R"],
                             "model_expected": min(lA, lR, r["L_R"] * lH(p, a, hh)) + spend * r["L_R"]})
                if pol == "DP" and mode == "indep":
                    # Proposition 1 certificate at the stopping state (independence: TV_T <= min(1, sum TV_k))
                    rem = [k for k in range(ev.K) if k not in calls]
                    if rem:
                        g, _ = g_and_kinks(a, hh); tvT = min(1.0, sum(ev.TV[k] for k in rem))
                        zi = np.log(p / (1 - p)); vz = True
                        for k in rem:   # one-step VOI of each remaining experiment (Theorem 1(iii): zero iff no kink crossing)
                            m = p * ev.f1[k] + (1 - p) * ev.f0[k]; pn = sig(zi + ev.w[k])
                            if g(p) - (m * g(pn)).sum() > 1e-12: vz = False
                        cert.append({"cost": cost, "certificate_holds": c.min() >= min(g(p), (a + 1) * p * (1 - p) * tvT),
                                     "all_one_step_voi_zero": vz})
        print(f"cost {cost} done, {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows); res.to_csv(OUT / "case_results.csv", index=False)
    summ = res.groupby(["cost", "policy"]).agg(TDC=("total", "mean"), loss=("loss", "mean"), spend=("spend", "mean"),
                                                 model_expected=("model_expected", "mean"), calls=("calls", "mean"),
                                                 defer=("action", lambda s: (s == "H").mean()), mean_p=("p", "mean"),
                                                 rate=("theta", "mean"), mean_prior=("prior", "mean")).reset_index()
    summ["optimism_gap"] = summ["TDC"] - summ["model_expected"]
    summ = summ.merge(res.assign(se=(res["p"] - res["theta"]) ** 2).groupby(["cost", "policy"])["se"].mean().rename("brier").reset_index(), on=["cost", "policy"])
    summ.to_csv(OUT / "policy_summary.csv", index=False)
    # paired bootstrap: ALL minus DP (indep) realised saving, versus sum of costs
    boot = []
    for cost in sorted(res["cost"].unique()):
        A = res[(res.cost == cost) & (res.policy == "ALL")]["total"].values
        P = res[(res.cost == cost) & (res.policy == "DP")]["total"].values
        d = A - P; n = len(d)
        bs = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(1000)])
        boot.append({"cost": cost, "sum_c": 4 * cost, "saving_mean": d.mean(), "ci_lo": np.quantile(bs, .025), "ci_hi": np.quantile(bs, .975),
                     "TDC_ALL": A.mean(), "ceiling_share": 4 * cost / A.mean()})
    pd.DataFrame(boot).to_csv(OUT / "savings_vs_ceiling.csv", index=False)
    cdf = pd.DataFrame(cert)
    if len(cdf): cdf.groupby("cost")[["certificate_holds", "all_one_step_voi_zero"]].mean().to_csv(OUT / "certificate_share.csv")
    json.dump({"eta": ETA, "eps": EPS, "n_test_used": int(len(T)), "r_hat": ev.r, "a_median": a_med, "best_single": ev.names[best_single],
               "seconds": time.time() - t0}, open(OUT / "meta.json", "w"), indent=2)
    print("done", time.time() - t0)

if __name__ == "__main__":
    main()
