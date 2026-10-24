# When to ask another agent

Code written by Narayana Darapaneni.

Code and logged outputs for the article *When to ask another agent: Value-of-information orchestration of AI agents and human review in credit decisions*.

The article treats each AI agent as a costly source of evidence about default risk. After each answer the lender approves, rejects, refers the file to a human analyst or consults another agent. This repository holds the code for the theory checks, the experiments with statistical evidence sources on two public credit files, and the experiments with nineteen language models.

## Contents

| Folder | What it holds | Article |
|---|---|---|
| `theory/` | Algorithms 1 and 2 with a numerical check of Proposition 4; the optimal regions of the worked example; Figure 1 | Sections 2–3, Fig. 1 |
| `statistical_sources/` | The shared module `lc_verify.py` (sample, losses, sources, policies), combination rules, the copula policy, unequal prices, borrower groups, the second portfolio and the price comparison | Sections 4.1–4.5, 5.3 |
| `llm_agents/` | Prompts and calls of the language-model agents, the logged answers of the run of 4 October 2026, the analysis and its result files | Section 4.6 |
| `paper_numbers/` | Collects every number of Section 4.6 from the result files into one JSON file | Section 4.6 |
| `data/` | Where the two data files go, with their fingerprints | Section 4.1 |

## Requirements

Python 3.9 or newer.

    pip install -r requirements.txt

## Data

Both data files are public and are not included here. See `data/README.md` for the sources and the SHA-256 fingerprint of the loan file used in the article.

## Reproducing the results

### Theory (no data needed)

    cd theory
    python fig1_value_of_consultation.py   # Figure 1; prints TV = 0.383 and the values at p = 0.10, 0.20, 0.35
    python example_thresholds.py           # optimal regions of the worked example with one agent
    python algorithms.py                   # Algorithms 1 and 2; numerical check of Proposition 4

### Statistical evidence sources

The scripts in `statistical_sources/` import `lc_verify.py` (see below).

    cd statistical_sources
    python dss_extensions.py 1.0 0.0       # main setting (no referral)
    python dss_extensions.py 0.5 0.1       # referral setting: review at half the limit, analyst error 0.1
    python taiwan_check.py                 # second portfolio (credit-card accounts)
    python price_differential.py RESULT_DIR

`dss_extensions.py` takes about 17 minutes per setting on two cores. Setting the environment variable `NSUB=200` runs it on a subsample of test loans and writes to a separate folder.

### Language-model agents

The logged answers (`llm_agents/llm_outputs.csv`, 15,660 rows) are the record of the experiment. No API key is needed to repeat the analysis:

    cd llm_agents
    python analyse_llm_agents.py --all
    python analyse_llm_agents.py --all --eta 0.5 --eps 0.1
    python compare_experiments.py
    python compare_experiments.py --referral
    cd ../paper_numbers
    python llm_facts.py

The full analysis takes about 40 minutes per setting on two cores, mostly for bootstrap intervals. The result files in `llm_agents/results_llm/` were written by an earlier version of `analyse_llm_agents.py`. Running the commands above regenerates them and adds `auc_diffs.csv` and `paired_diffs_cal.csv`, which the article uses for the within-block AUC comparisons and for the intervals that redraw the calibration loans.

To collect new answers, copy `.env.example` to `.env`, add an OpenAI API key, and run

    python check_setup.py
    python run_llm_agents.py --plan
    python run_llm_agents.py --experiment probe

and then the experiments `survey`, `teams` and `effort`. A new run will not reproduce the logged scores exactly: the reasoning models are not reproducible call by call, and the provider retires four of the models (GPT-5, GPT-5 Mini, GPT-5 nano and o3) on 11 December 2026. The total spend is capped by `total_budget_usd` in `experiments.json`.

## Shared module

`lc_verify.py` defines the sample, the blocks, the losses, the four evidence sources, the dynamic programme and the policies. Run as a script it writes the sample description, the source diagnostics and the policy results of Sections 4.1–4.4 (Tables 2, 3 and 5 of the article):

    cd statistical_sources
    python lc_verify.py 1.0 0.0
    python lc_verify.py 0.5 0.1

The same file is in `llm_agents/`, because the language-model scripts use its sample and losses.

## Author

Narayana Darapaneni wrote the code in this repository and ran the experiments.

## Licence

MIT. See `LICENSE`. If you use the code, please cite the article (see `CITATION.cff`).
