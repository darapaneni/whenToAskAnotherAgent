"""Run language-model agents on the Lending Club sample and log their outputs.

Each agent is a (role, model) pair: the role decides what part of a loan application the model reads, and the
model returns a risk score from 0 to 100. Outputs go to llm_outputs.csv, one row per (loan, agent), with token
counts (reasoning tokens included), latency and the raw answer. Experiments are defined in experiments.json and
models in models.json. A call that is already in llm_outputs.csv is never repeated, so experiments share calls
and a run can be stopped and restarted at any time.

    python run_llm_agents.py --list                          # the experiments and their budgets
    python run_llm_agents.py --plan                          # no API call: cost forecast against the total budget
    python run_llm_agents.py --experiment probe              # every model on 8 loans
    python run_llm_agents.py --experiment survey             # the model survey
    python run_llm_agents.py --status                        # what has been spent and what is left
    python run_llm_agents.py --experiment survey --mock      # plumbing test with a fake model; NOT a result

The total spend is capped (total_budget_usd in experiments.json): a call is started only if its worst-case cost
still fits, so the total recorded here, at the prices in models.json, stays below the cap. An attempt that ends
without an answer (timeout, lost connection) is counted at its worst case. What the code cannot see - other use
of the same key, prices that differ from models.json, taxes - is outside the cap; prepaying the budget with
automatic recharge switched off is the guarantee on the provider's side.

API keys are read from the file .env in this folder (copy .env.example to .env and paste the key).

Author: Narayana Darapaneni.
"""
import argparse, csv, hashlib, json, os, re, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import voi_common as C

HERE = Path(__file__).parent
C.load_env()
_argv, sys.argv = sys.argv, [sys.argv[0], "1.0", "0.0"]      # lc_verify reads its own arguments at import
import lc_verify as V                                         # noqa: E402  (sample, blocks, losses)
sys.argv = _argv

SYSTEM = ("You are a credit analyst at a consumer lender. You assess applications for 36-month unsecured personal "
          "loans. Judge only from the information given. Reply with a single JSON object and nothing else.")

TASK = {
    "text": ("Below is what the borrower wrote when applying: the loan title, the stated purpose and, if present, a "
             "free-text description. Using only this text, rate the risk that the borrower will default on the loan."),
    "file": ("Below are the structured fields of the application as known at origination. Using only these fields, "
             "rate the risk that the borrower will default on the loan."),
    "policy": ("Below are the lender's written credit policy and the structured fields of an application. Check the "
               "application against the policy and rate the risk that the borrower will default on the loan. "
               "List the numbers of the policy rules the application breaches."),
}
FORMAT = {
    "text": '{"risk_score": <integer 0-100, 0 = certain to repay, 100 = certain to default>, "reason": "<at most 20 words>"}',
    "file": '{"risk_score": <integer 0-100, 0 = certain to repay, 100 = certain to default>, "reason": "<at most 20 words>"}',
    "policy": '{"risk_score": <integer 0-100, 0 = certain to repay, 100 = certain to default>, "rules_breached": [<rule numbers>], "reason": "<at most 20 words>"}',
}

# Fields an agent may see. Excluded on purpose: the platform's grade, sub-grade and interest rate (outputs of its
# own risk model), everything known only after origination, ZIP code and state.
FILE_FIELDS = [("loan_amnt", "Loan amount requested (USD)"), ("purpose", "Purpose"), ("emp_title", "Employer or job title"),
               ("emp_length", "Employment length"), ("home_ownership", "Home ownership"),
               ("annual_inc", "Annual income (USD)"), ("verification_status", "Income verification"),
               ("dti", "Debt-to-income ratio (%)"), ("delinq_2yrs", "Delinquencies in the last 2 years"),
               ("earliest_cr_line", "Earliest credit line"), ("inq_last_6mths", "Credit inquiries in the last 6 months"),
               ("mths_since_last_delinq", "Months since last delinquency"), ("open_acc", "Open credit accounts"),
               ("pub_rec", "Derogatory public records"), ("revol_bal", "Revolving balance (USD)"),
               ("revol_util", "Revolving utilisation"), ("total_acc", "Total credit accounts")]


def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"Borrower added on \S+ >", " ", t); t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def render(row, role, policy_text):
    if role == "text":
        desc = clean_text(row.get("desc")) or "(no description given)"
        return f"Loan title: {clean_text(row.get('title')) or '(none)'}\nPurpose: {row.get('purpose')}\nDescription: {desc[:3000]}"
    lines = []
    for col, label in FILE_FIELDS:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = "not reported"
        lines.append(f"{label}: {v}")
    lines.append(f"Application date: {row['date'].strftime('%B %Y')}")
    fields = "\n".join(lines)
    return fields if role == "file" else f"CREDIT POLICY\n{policy_text}\n\nAPPLICATION\n{fields}"


def messages(row, role, policy_text):
    user = f"{TASK[role]}\n\n{render(row, role, policy_text)}\n\nAnswer in this JSON format:\n{FORMAT[role]}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_score(text):
    try:
        s = json.loads(text).get("risk_score")
    except Exception:
        m = re.search(r'"risk_score"\s*:\s*(-?\d+(?:\.\d+)?)', text or "")
        s = m.group(1) if m else None
    try:
        s = float(s)
    except (TypeError, ValueError):
        return np.nan
    return s if 0 <= s <= 100 else np.nan


class Blocked(Exception):
    """An error that will not go away by retrying: wrong key, no credit, unknown model, daily request cap."""
    def __init__(self, scope, reason):
        super().__init__(reason); self.scope, self.reason = scope, reason     # scope: "provider" or "model"


def classify(e, prov, model):
    """Return a Blocked exception if the error is permanent for this run, else None."""
    name, msg = type(e).__name__, str(e)
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return Blocked("provider", f"the API key for provider '{prov}' was rejected. Check the key in the file .env.")
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return Blocked("provider", f"the account of provider '{prov}' has no credit left. Add credit on the provider's billing page.")
    if name == "NotFoundError" or "model_not_found" in msg or "does not exist" in msg:
        return Blocked("model", f"model '{model}' is not available to this key (retired, renamed, or the account lacks access).")
    if name == "RateLimitError" and re.search(r"per day|requests per day|\bRPD\b|daily", msg, re.I):
        return Blocked("model", f"the daily request limit for '{model}' is used up. Run the same command again tomorrow; finished calls are kept.")
    return None


MOCK_REASONING = {None: 0, "none": 0, "minimal": 30, "low": 150, "medium": 500, "high": 1500, "xhigh": 3000, "max": 5000}


def _unit(*parts):
    """Deterministic standard-normal number from a hash of the parts (for the fake model only)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    u1 = (int.from_bytes(h[:8], "big") + 1) / (2 ** 64 + 2); u2 = int.from_bytes(h[8:16], "big") / 2 ** 64
    return float(np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))


class MockClient:
    """Deterministic fake model for testing the plumbing without an API key. Its outputs are NOT results.
    Scores carry a weak signal plus noise shared within a role and within a model class, so that the analysis has
    informativeness, dependence and prices to work with."""
    def __init__(self, reg):
        self.reg = reg

    def call(self, role, model, effort, msgs, row):
        spec = self.reg["models"][model]; cls = spec["class"]
        skill = 0.35 + 0.45 * (int(hashlib.sha256(model.encode()).hexdigest()[:6], 16) / 0xFFFFFF)
        skill += {"none": -0.1, "minimal": -0.05, "low": 0.0, "medium": 0.05, "high": 0.08}.get(effort, 0.0)
        z = (skill * 0.9 * (float(row["theta"]) - 0.11) + 0.55 * _unit(row["id"], "role", role) + 0.45 * _unit(row["id"], "class", cls)
             + 0.75 * _unit(row["id"], role, model, effort))
        text = json.dumps({"risk_score": int(np.clip(round(45 + 16 * z), 0, 100)), "reason": "mock output"})
        n_in = sum(len(m["content"]) for m in msgs) // 4
        n_reason = MOCK_REASONING.get(effort, 0) * (3 if spec.get("pro") else 1)
        return {"text": text, "input_tokens": n_in, "output_tokens": 25 + n_reason, "reasoning_tokens": n_reason,
                "status": "completed", "model_returned": "mock", "effort_used": effort}


class OpenAIClient:
    """Calls a model through Chat Completions ('api': 'chat') or the Responses API ('api': 'responses').
    Parameters a model rejects are dropped once and remembered, and the change is printed."""
    def __init__(self, reg):
        from openai import OpenAI
        self.reg, self.clients, self.adapt, self.lock = reg, {}, {}, threading.Lock()
        self.on_lost = None                              # called when an attempt ends without an answer (it may still be billed)
        for name, p in reg["providers"].items():
            key = os.environ.get(p["api_key_env"]) if p.get("api_key_env") else "not-needed"
            if key:
                self.clients[name] = OpenAI(api_key=key, base_url=p.get("base_url") or None, max_retries=0, timeout=600)

    def _note(self, model, what):
        with self.lock:
            a = self.adapt.setdefault(model, set())
            if what not in a:
                a.add(what); print(f"  note: {model}: {what}", flush=True)

    def _has(self, model, what):
        return what in self.adapt.get(model, ())

    def _once(self, client, role, model, effort, msgs):
        spec = self.reg["models"][model]; api = "responses" if self._has(model, "switched to the Responses API") else spec.get("api", "chat")
        use_effort = None if (effort is None or self._has(model, "reasoning effort not accepted; provider default used")) else effort
        budget = C.output_budget(model, effort, self.reg)                # room for the answer and, where it applies, the reasoning
        json_mode = not self._has(model, "JSON mode not accepted; plain text parsed instead")
        if api == "responses":
            kw = dict(model=model, instructions=msgs[0]["content"], input=msgs[1]["content"], max_output_tokens=budget)
            if use_effort is not None:
                kw["reasoning"] = {"effort": use_effort}
            if json_mode:
                kw["text"] = {"format": {"type": "json_object"}}
            r = client.responses.create(**kw); u = r.usage
            det = getattr(u, "output_tokens_details", None)
            return {"text": getattr(r, "output_text", "") or "", "input_tokens": getattr(u, "input_tokens", 0),
                    "output_tokens": getattr(u, "output_tokens", 0), "reasoning_tokens": getattr(det, "reasoning_tokens", 0) or 0,
                    "status": getattr(r, "status", "") or "", "model_returned": getattr(r, "model", model), "effort_used": use_effort}
        kw = dict(model=model, messages=msgs)
        if not self._has(model, "temperature not accepted; provider default used"):
            kw["temperature"] = 0
        if not self._has(model, "seed not accepted"):
            kw["seed"] = 20260929
        kw["max_completion_tokens" if self._has(model, "max_completion_tokens used") or use_effort is not None else "max_tokens"] = budget
        if use_effort is not None:
            kw["reasoning_effort"] = use_effort
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kw); u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        return {"text": r.choices[0].message.content or "", "input_tokens": getattr(u, "prompt_tokens", 0),
                "output_tokens": getattr(u, "completion_tokens", 0), "reasoning_tokens": getattr(det, "reasoning_tokens", 0) or 0,
                "status": r.choices[0].finish_reason or "", "model_returned": getattr(r, "model", model), "effort_used": use_effort}

    def call(self, role, model, effort, msgs, row):
        prov = self.reg["models"][model].get("provider", "openai")
        if prov not in self.clients:
            raise Blocked("provider", f"no API key found for provider '{prov}'. Put it in the file .env (see .env.example).")
        waits = 0
        for attempt in range(14):
            try:
                return self._once(self.clients[prov], role, model, effort, msgs)
            except Exception as e:
                msg = str(e); low = msg.lower(); blocked = classify(e, prov, model)
                if blocked:
                    raise blocked from None
                if type(e).__name__ in ("BadRequestError", "UnprocessableEntityError") or "unsupported" in low or "not supported" in low:
                    if "v1/responses" in low and not self._has(model, "switched to the Responses API"):
                        self._note(model, "switched to the Responses API"); continue
                    if "max_tokens" in low and "max_completion_tokens" in low and not self._has(model, "max_completion_tokens used"):
                        self._note(model, "max_completion_tokens used"); continue
                    if "temperature" in low and not self._has(model, "temperature not accepted; provider default used"):
                        self._note(model, "temperature not accepted; provider default used"); continue
                    if "seed" in low and not self._has(model, "seed not accepted"):
                        self._note(model, "seed not accepted"); continue
                    if ("reasoning" in low or "effort" in low) and not self._has(model, "reasoning effort not accepted; provider default used"):
                        self._note(model, "reasoning effort not accepted; provider default used"); continue
                    if ("json" in low or "response_format" in low or "text.format" in low) and not self._has(model, "JSON mode not accepted; plain text parsed instead"):
                        self._note(model, "JSON mode not accepted; plain text parsed instead"); continue
                    raise
                kind = type(e).__name__.lower()
                if self.on_lost and ("timeout" in kind or "connection" in kind) and not self.on_lost(role, model, effort, msgs):
                    raise                                                    # no money left for another attempt
                waits += 1
                if waits > 7:
                    raise
                time.sleep(min(60, 2 ** waits + 0.5))                        # rate limits, timeouts: back off and retry
        raise RuntimeError(f"{model}: the request was rejected after every adjustment")


def tokens_in(role, df, order, policy_text):
    sample = [df.loc[i] for i in order[:: max(1, len(order) // 40)]][:40]
    return float(np.mean([sum(len(m["content"]) for m in messages(r, role, policy_text)) / 4 for r in sample])) if sample else 0.0


def estimate(agents, member, df, order, reg, store, policy_text):
    """Cost forecast per agent: measured tokens per call where the store has them, assumptions otherwise.
    Returns rows (agent, loans, calls still to make, tokens in, tokens out, cost to come, basis) and the total."""
    rows, total, tin = [], 0.0, {}
    for key, role, model, eff, _, _ in agents:
        prev = store[(store["agent"] == key) & store["risk_score"].notna()]
        if role not in tin:
            tin[role] = tokens_in(role, df, order, policy_text)
        if len(prev) >= 4:
            n_in, n_out, basis = prev["input_tokens"].mean(), prev["output_tokens"].mean(), f"measured on {len(prev)} calls"
        else:
            n_in = tin[role]; n_out = 40 + MOCK_REASONING.get(eff, 0)
            basis = "assumed" + (" (reasoning tokens are a guess until the probe has run)" if eff not in (None, "none") else "")
        done = set(store.loc[store["agent"] == key, "loan_id"])
        todo = sum(1 for i in member[key] if df.at[i, "id"] not in done)
        cost = todo * C.call_cost(model, n_in, n_out, reg); total += 0 if np.isnan(cost) else cost
        rows.append((key, len(member[key]), todo, n_in, n_out, cost, basis))
    return rows, total, tin


def status(reg, exps, df, store, label):
    cap = exps["total_budget_usd"]; real = label == C.STORE.name; spent = C.total_spent(store, ledger=real)
    if not real:
        print("FAKE MODEL: these are calls of the built-in fake model, for testing only. No money was spent.")
    print(f"Spending so far ({label}): USD {spent:.2f} of the total budget of USD {cap:.2f}; USD {max(cap - spent, 0):.2f} left.")
    if real and C.ledger_total():
        print(f"  of which USD {C.ledger_total():.2f} in spend_ledger.csv (setup test calls; attempts without an answer, at their worst case)")
    if C.unpriced(store):
        print("  WARNING: no price in models.json for " + ", ".join(C.unpriced(store)) + " - the calls of these models are NOT in the total. "
              "Do not delete a model from \"models\" in models.json; remove it from the lists under \"groups\" instead.")
    if len(store):
        t = store.groupby("model").agg(calls=("risk_score", "size"), usd=("cost_usd", "sum")).sort_values("usd", ascending=False)
        print(t.round(3).to_string())
    for name, exp in exps["experiments"].items():
        agents, order, member = C.plan(exp, reg, df)
        done = sum(int(store.loc[store["agent"] == g[0], "loan_id"].isin(set(df["id"].loc[list(member[g[0]])])).sum()) for g in agents) if len(store) else 0
        print(f"  {name:8s} {done:6d} of {sum(len(member[g[0]]) for g in agents):6d} calls done; USD {C.experiment_spent(store, agents, member, df):.2f} of its budget of USD {exp['budget_usd']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", "-e", default=None)
    ap.add_argument("--list", action="store_true", help="list the experiments and exit")
    ap.add_argument("--status", action="store_true", help="show what has been spent and what is left, and exit")
    ap.add_argument("--plan", action="store_true", help="cost forecast for every experiment against the total budget; no API call")
    ap.add_argument("--budget", type=float, default=None, help="lower the spending limit of this run (USD); it cannot raise the limits in experiments.json")
    ap.add_argument("--workers", type=int, default=4, help="parallel calls; 4 suits a new account, 16 a higher usage tier")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    reg, exps = C.registry(), C.experiments(); cap = float(exps["total_budget_usd"])
    if a.list or not (a.experiment or a.status or a.plan):
        print(f"Experiments (run with --experiment NAME). Total budget: USD {cap:.2f}.")
        for name, e in exps["experiments"].items():
            print(f"  {name:8s} {e['title']}  [up to USD {e['budget_usd']:.2f}]")
        return
    policy_text = (HERE / "credit_policy.md").read_text(encoding="utf-8")
    out = HERE / "llm_outputs_MOCK.csv" if a.mock else C.STORE
    df = V.load(); df["id"] = df["id"].astype(str)
    store = C.read_store(out, reg); real = not a.mock
    if a.status:
        status(reg, exps, df, store, out.name); return
    if a.plan:
        spent = C.total_spent(store, ledger=real); union, per_call = {}, {}
        if a.mock:
            print("FAKE MODEL: token counts marked 'measured' below come from the built-in fake model, not from a provider.")
        print(f"Plan against the total budget of USD {cap:.2f} (spent so far: USD {spent:.2f})")
        for name, exp in exps["experiments"].items():
            agents, order, member = C.plan(exp, reg, df); rows, est, _ = estimate(agents, member, df, order, reg, store, policy_text)
            measured = sum(1 for r in rows if r[6].startswith("measured"))
            for g, r in zip(agents, rows):
                union.setdefault(g[0], set()).update(member[g[0]]); per_call[g[0]] = C.call_cost(g[2], r[3], r[4], reg)
            print(f"  {name:8s} calls to make {sum(r[2] for r in rows):6d}   estimated USD {est:6.2f}   budget USD {exp['budget_usd']:.2f}   "
                  f"({measured} of {len(rows)} agents with measured tokens)")
            left = min(exp["budget_usd"] - C.experiment_spent(store, agents, member, df), cap - spent)
            if est > left + 1e-9:                        # say which blocks of calls the money reaches
                blk = C.agent_blocks(exp, reg); parts = []
                for b, block in enumerate(exp["calls"]):
                    need = sum(0 if np.isnan(r[5]) else r[5] for g, r in zip(agents, rows) if blk[g[0]] == b)
                    share = 1.0 if need <= 0 else min(1.0, max(left, 0) / need); left -= need * share
                    what = block["models"] if isinstance(block["models"], str) else "+".join(block["models"])
                    parts.append(f"{what}: {'all' if share >= 0.999 else 'none' if share < 0.005 else f'about {100 * share:.0f}%'}")
                print("           more than this experiment may spend. Calls are made block by block, so the money reaches - " + "; ".join(parts))
        done = set(zip(store["loan_id"], store["agent"]))
        tot = sum(per_call[k] * sum(1 for i in loans if (df.at[i, "id"], k) not in done) for k, loans in union.items())
        print(f"  estimated cost of everything still to run, each call counted once: USD {tot:.2f}; with what is spent: USD {spent + tot:.2f}")
        print("  " + ("This fits the budget." if spent + tot <= cap else
                      "This is more than the budget: the runs will stop when the money is used up. Within an experiment the calls are made "
                      "block by block in the order of experiments.json (cheapest models first), so the dearest models lose loans first. "
                      "To choose differently, lower n_cal and n_test of a block in experiments.json before running."))
        return

    exp = C.get_experiment(a.experiment, exps)
    agents, order, member = C.plan(exp, reg, df)
    print(f"experiment '{a.experiment}': {exp['title']}")
    for (nc, nt), ks in sorted({(g[4], g[5]): [h[0] for h in agents if (h[4], h[5]) == (g[4], g[5])] for g in agents}.items(), reverse=True):
        print(f"  {len(ks):2d} agents on {nc} calibration loans (half defaulted, half repaid) and {nt} test loans")

    done = set(zip(store["loan_id"], store["agent"]))
    blk = C.agent_blocks(exp, reg)                       # block by block (cheapest models first), loans in sample order within a block
    todo = [(i, g) for b in sorted(set(blk.values())) for i in order for g in agents
            if blk[g[0]] == b and i in member[g[0]] and (df.at[i, "id"], g[0]) not in done]
    n_all = sum(len(member[g[0]]) for g in agents)
    print(f"calls to make: {len(todo)} (already in {out.name}: {n_all - len(todo)})")

    rows, est, tin = estimate(agents, member, df, order, reg, store, policy_text)
    spent_total = C.total_spent(store, ledger=real); spent_exp = C.experiment_spent(store, agents, member, df)
    limit = min(cap - spent_total, exp["budget_usd"] - spent_exp, a.budget if a.budget is not None else float("inf"))
    if a.dry_run:
        for key, n, td, n_in, n_out, cost, basis in rows:
            print(f"  {key:30s} loans {n:5d}  calls left {td:5d}  ~{n_in:4.0f} in + {n_out:4.0f} out tokens  est. USD {cost:6.2f}   [{basis}]")
    print(f"estimated cost of the remaining calls: USD {est:.2f}")
    print(f"spent so far: USD {spent_total:.2f} of USD {cap:.2f} in total, USD {spent_exp:.2f} of USD {exp['budget_usd']:.2f} for this experiment; "
          f"this run stops at USD {max(limit, 0):.2f}")
    if a.dry_run:
        print("prices are those in models.json - check them against the provider's price page"); return
    if limit <= 0:
        print("Nothing is left to spend for this experiment. See python run_llm_agents.py --status"); return

    lockfile = HERE / ("run_MOCK.lock" if a.mock else "run.lock")      # two runs at once would each see only their own spending
    try:
        os.close(os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        sys.exit(f"Another run seems to be in progress ({lockfile.name} exists). Wait for it to finish. "
                 f"If no run is in progress (for example after a crash), delete the file {lockfile.name} and start again.")
    try:
        _run(a, reg, exp, agents, member, todo, tin, limit, cap, out, df, policy_text, real)
    finally:
        lockfile.unlink(missing_ok=True)


def _run(a, reg, exp, agents, member, todo, tin, limit, cap, out, df, policy_text, real):
    client = MockClient(reg) if a.mock else OpenAIClient(reg)
    lock = threading.Lock(); new = not out.exists()
    fh = open(out, "a", newline="", encoding="utf-8"); w = csv.writer(fh)
    if new:
        w.writerow(C.STORE_COLS)
    blocked, stop, over, spent, held, lost = {}, threading.Event(), threading.Event(), [0.0], [0.0], [0.0]

    def n_in_of(msgs):
        return sum(len(m["content"]) for m in msgs) / 4

    def on_lost(role, model, eff, msgs):                 # an attempt without an answer may still be billed: count its worst case
        c = C.worst_case_cost(model, eff, reg, n_in_of(msgs))
        with lock:
            spent[0] += c; lost[0] += c; room = spent[0] + held[0] <= limit
            if real:
                C.ledger_add(f"attempt without an answer ({model}) counted at its worst case", c)
        if not room:
            stop.set(); over.set()
        return room                                      # False: no further attempt for this call

    if real:
        client.on_lost = on_lost

    def is_blocked(model):
        return ("provider", reg["models"][model].get("provider", "openai")) in blocked or ("model", model) in blocked

    def work(i, g):
        key, role, model, eff = g[:4]
        if stop.is_set() or is_blocked(model):
            return None
        row = df.loc[i]; msgs = messages(row, role, policy_text)
        worst = C.worst_case_cost(model, eff, reg, n_in_of(msgs))
        with lock:                                       # a call starts only if its worst case still fits the limit
            if spent[0] + held[0] + worst > limit:
                stop.set(); over.set(); return None
            held[0] += worst
        try:
            t0 = time.time()
            r = client.call(role, model, eff, msgs, row)
            if not (r["input_tokens"] or r["output_tokens"]):        # no usage reported: record the worst case, never zero
                r["input_tokens"] = int(np.ceil(C.INPUT_MARGIN * n_in_of(msgs))); r["output_tokens"] = C.output_budget(model, eff, reg)
                r["status"] = (r["status"] + "|usage_missing").strip("|")
            cost = C.call_cost(model, r["input_tokens"], r["output_tokens"], reg)
            with lock:
                spent[0] += 0 if np.isnan(cost) else cost
        finally:
            with lock:
                held[0] -= worst
        return [row["id"], row["block"], key, role, model, r["effort_used"] or "", parse_score(r["text"]), r["input_tokens"], r["output_tokens"],
                r["reasoning_tokens"], round(time.time() - t0, 3), r["status"], (r["text"] or "").replace("\n", " ")[:400], r["model_returned"],
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")]

    n = failed = streak = 0; t0 = time.time(); last_err = ""; announced = []

    def on_interrupt(signum, frame):                     # Ctrl+C: start no new call, wait for those already sent, save them
        if stop.is_set():
            raise KeyboardInterrupt                      # a second Ctrl+C ends the run at once
        stop.set(); print("interrupted - waiting for the calls already sent, which are saved; run the same command to continue", flush=True)

    old_handler = signal.signal(signal.SIGINT, on_interrupt)
    ex = ThreadPoolExecutor(max_workers=a.workers)
    futs = {ex.submit(work, i, g): g for i, g in todo}
    try:
        for f in as_completed(futs):
            g = futs[f]
            try:
                rec = f.result()
            except Blocked as b:
                bkey = (b.scope, reg["models"][g[2]].get("provider", "openai") if b.scope == "provider" else g[2])
                if bkey not in blocked:
                    blocked[bkey] = b.reason; print("STOPPED for", bkey[1] + ":", b.reason, flush=True)
                continue
            except Exception as e:
                failed += 1; streak += 1; last_err = f"{g[0]}: {repr(e)[:300]}"
                if streak >= 25:
                    stop.set(); print("25 calls in a row failed; stopping. Last error:", last_err, flush=True)
                continue
            if over.is_set() and not announced:
                announced.append(1); print(f"spending limit of this run (USD {limit:.2f}) reached; stopping.", flush=True)
            if rec is None:
                continue
            streak = 0
            with lock:
                w.writerow(rec); fh.flush(); n += 1      # every answer is on disk at once: a paid call is never lost
                if n % 100 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} calls, USD {spent[0]:.2f} spent in this run, {el / 60:.1f} min elapsed", flush=True)
    except KeyboardInterrupt:
        stop.set(); print("stopped at once; an answer that had not yet arrived is not saved", flush=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True); fh.close(); signal.signal(signal.SIGINT, old_handler)

    res = C.read_store(out, reg); sub = res[res["agent"].isin([g[0] for g in agents])]
    left = len(todo) - n
    if a.mock:
        print("\nFAKE MODEL: the rows below come from the built-in fake model, for testing only. No money was spent.")
    if lost[0]:
        print(f"\nUSD {lost[0]:.2f} of this run is for attempts that ended without an answer, counted at their worst case (spend_ledger.csv).")
    print(f"\n{out.name}: {len(res)} rows in total ({n} new, USD {spent[0]:.2f} in this run); "
          f"total spent USD {C.total_spent(res, ledger=real):.2f} of USD {cap:.2f}")
    if len(sub):
        t = sub.groupby("agent").agg(calls=("risk_score", "size"), unparsed=("risk_score", lambda s: int(s.isna().sum())),
                                     tokens_in=("input_tokens", "mean"), tokens_out=("output_tokens", "mean"),
                                     reasoning=("reasoning_tokens", "mean"), seconds=("latency_s", "mean"), usd_per_call=("cost_usd", "mean"))
        pd.set_option("display.width", 200); print(t.round({"tokens_in": 0, "tokens_out": 0, "reasoning": 0, "seconds": 1, "usd_per_call": 5}).to_string())
    if left:
        print(f"{left} calls were not made ({failed} failed after retries).")
        for (scope, name), reason in blocked.items():
            print(f"  - {name}: {reason}")
        if last_err and not blocked:
            print("  last error:", last_err)
        if over.is_set():
            print("The spending limit was reached. The analysis uses the loans that were completed; nothing needs to be repeated.")
        else:
            print("Run the same command again to continue; finished calls are skipped.")
    else:
        print(f"All calls of '{a.experiment}' are done. Next: python analyse_llm_agents.py --experiment {a.experiment}" + (" --mock" if a.mock else ""))


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is needed.")
    main()
