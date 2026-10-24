"""Step 1: check that everything is in place before any money is spent.

Checks Python and the packages, the data file, the model registry and the API key, asks the provider which of
the models this key can use, and then makes ONE small test call to each model of the chosen experiment.

    python check_setup.py              # checks the 19 models of the probe (about 5 cents, counted against the total budget)
    python check_setup.py --no-call    # everything except the test calls

Author: Narayana Darapaneni.
"""
import argparse, hashlib, importlib, os, sys
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--experiment", "-e", default="probe")
ap.add_argument("--no-call", action="store_true")
a = ap.parse_args()
ok = True


def line(good, text, warn=False):
    global ok
    ok = ok and (bool(good) or warn)
    print(("  [ok]   " if good else "  [note] " if warn else "  [FAIL] ") + text)


print("1. Python and packages")
line(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]} (3.9 or newer is needed)")
for mod, pip_name in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("openai", "openai")):
    try:
        m = importlib.import_module(mod); line(True, f"{pip_name} {getattr(m, '__version__', '')}")
    except Exception:
        line(False, f"{pip_name} is not installed  ->  pip install -r requirements.txt")
if not ok:
    sys.exit("\nInstall the missing packages (see README.md) and run this check again.")

print("2. Data file")
data = HERE / "data" / "LoanStats3a.csv"
if data.exists():
    want = (HERE / "data_sha256.txt").read_text().split()[0]
    got = hashlib.sha256(data.read_bytes()).hexdigest()
    line(got == want, "data/LoanStats3a.csv found" + (" and identical to the file used in the paper" if got == want else " but it is NOT the file used in the paper (fingerprint differs)"))
else:
    line(False, "data/LoanStats3a.csv is missing - copy the loan file into llm_agents/data/ (see data/README.md)")

print("3. Registry, experiment and key")
import voi_common as C                                       # noqa: E402
import run_llm_agents as R                                   # also reads .env  # noqa: E402
reg, exps = C.registry(), C.experiments(); exp = C.get_experiment(a.experiment, exps)
agents = C.experiment_agents(exp, reg); models = list(dict.fromkeys(g[2] for g in agents))
cap = float(exps["total_budget_usd"]); spent_before = C.total_spent(C.read_store(C.STORE, reg))
line((HERE / ".env").exists(), "file .env found" if (HERE / ".env").exists() else "no file .env - copy .env.example to .env and paste your key into it")
for prov in sorted({reg["models"][m].get("provider", "openai") for m in models}):
    env = reg["providers"][prov].get("api_key_env"); key = os.environ.get(env, "") if env else "x"
    line(bool(key), f"provider '{prov}': key {env} " + (f"found (ends in ...{key[-4:]})" if key else "is empty - paste it into .env"))
print(f"   experiment '{a.experiment}': {len(agents)} agents on {len(models)} models; prices in models.json were checked on {reg.get('prices_checked_on')}")
print(f"   total budget USD {cap:.2f}; spent so far USD {spent_before:.2f}")
soon = [m for m in models if "deprecated" in reg["models"][m].get("note", "")]
if soon:
    line(False, "the provider has announced the retirement of: " + ", ".join(soon) + " (see the notes in models.json) - run these before the shutdown date", warn=True)

available = None
if ok and not a.no_call:
    print("4. Models this key can use")
    try:
        client = R.OpenAIClient(reg)
        available = {m.id for m in client.clients["openai"].models.list()}
        missing = [m for m in models if m not in available]
        line(not missing, f"{len(models) - len(missing)} of {len(models)} models are listed for this key" + ("; NOT listed: " + ", ".join(missing) if missing else ""), warn=bool(missing))
    except Exception as e:
        line(False, f"could not list the models: {str(e)[:200]}", warn=True)

if a.no_call:
    print("5. Test calls skipped (--no-call)")
elif ok and cap - spent_before < 0.5:
    print("5. Test calls skipped: less than USD 0.50 of the total budget is left")
elif ok:
    print("5. One test call per model")
    import pandas as pd
    row = pd.Series({"id": "0", "theta": 0, "title": "Debt consolidation", "purpose": "debt_consolidation",
                     "desc": "I want to pay off two credit cards and close them. I have had the same job for six years.",
                     "loan_amnt": 8000, "emp_title": "Nurse", "emp_length": "6 years", "home_ownership": "RENT", "annual_inc": 52000,
                     "verification_status": "Verified", "dti": 14.2, "delinq_2yrs": 0, "earliest_cr_line": "Mar-2001",
                     "inq_last_6mths": 1, "mths_since_last_delinq": float("nan"), "open_acc": 7, "pub_rec": 0, "revol_bal": 9100,
                     "revol_util": "48%", "total_acc": 15, "date": pd.Timestamp("2011-05-01")})
    policy_text = (HERE / "credit_policy.md").read_text(encoding="utf-8"); seen, spent, bad = set(), 0.0, []
    for key, role, model, eff, _, _ in agents:
        if (model, eff) in seen:
            continue
        seen.add((model, eff))
        try:
            r = client.call(role, model, eff, R.messages(row, role, policy_text), row)
            score = R.parse_score(r["text"]); cost = C.call_cost(model, r["input_tokens"], r["output_tokens"], reg); spent += cost
            C.ledger_add(f"check_setup test call ({model})", cost)          # written at once, so an interrupted check is still counted
            good = score == score
            line(good, f"{key}: score {score if good else 'not readable'}; {r['input_tokens']} in + {r['output_tokens']} out tokens "
                       f"({r['reasoning_tokens']} reasoning), USD {cost:.4f}, reported as {r['model_returned']}"
                       + ("" if good else f"; answer was \"{(r['text'] or '').strip()[:80]}\" (status {r['status']})"))
            if not good:
                bad.append(model)
        except Exception as e:
            bad.append(model); line(False, f"{key}: {str(e)[:260]}")
    print(f"   test calls cost about USD {spent:.3f}; total spent so far USD {spent_before + spent:.2f} of USD {cap:.2f}")
    if bad:
        print("   To leave a model out, delete its name from the lists under \"groups\" in models.json (and from any team in experiments.json).")
else:
    print("5. Test calls skipped until the points marked [FAIL] are fixed")

print("\nAll checks passed. Next: python run_llm_agents.py --plan, then --experiment probe" if ok
      else "\nSome checks failed - fix the lines marked [FAIL] and run this check again.")
sys.exit(0 if ok else 1)
