# ICITIIT 2027 — Controlled ETL Benchmark

A Streamlit research dashboard for comparing a **rule-based ETL pipeline** with an **LLM-assisted ETL pipeline** on the same controlled input records. The application reports field-level accuracy, complete-record accuracy, observed processing time, throughput, token usage, API failures, and estimated model-service cost.

**Research paper title**

> AI-Augmented Marketing Analytics: Evaluating Large Language Models in ETL Pipelines for Accuracy, Cost, and Scalability in Small-Scale Deployments

This repository contains the experimental application and protocol. It does not contain research findings until an experiment has actually been run and the audit outputs reviewed.

## 1. Dataset source and scope

**Dataset:** [E-Commerce Returns: Who Will Send It Back? — Kaggle competition](https://www.kaggle.com/competitions/retail-return-risk-modeling)

The Kaggle listing describes transaction-level records for 2022–2024, with customer purchase history, product attributes, browsing/session features and shipping/delivery information. The target field `returned` belongs to the competition's return-prediction task.

For this study, the dashboard uses **`train.csv` only** as a source of records. `test.csv` has no observed `returned` target, and `sample_submission.csv` contains submission placeholders; neither is ground truth for this ETL experiment. The dashboard does **not** train or evaluate a return-prediction model.

**Important provenance caveat:** the Kaggle listing's date range and description are source claims. The originating retailer and the dataset's real-world provenance have not been independently verified. Therefore, the experiment must be described as a **controlled e-commerce dataset ETL benchmark**, not as a study of verified transactions collected from small businesses. Do not make unsupported claims about the dataset being real, representative, or produced by a particular retailer.

**Citation for the dataset listing:**

Hak. “E-Commerce Returns: Who Will Send It Back?” Kaggle competition, 2026. <https://www.kaggle.com/competitions/retail-return-risk-modeling>.

Check the Kaggle competition's current rules and data-use terms before publication or redistribution. This repository does not include the CSV data. Download the files through Kaggle using your own account and upload `train.csv` directly to the running Streamlit app.

For the file snapshot supplied while this application was prepared, `train.csv` contains 200,000 rows and 16 columns. Under the currently implemented eligibility rules, 78,964 rows (39.5%) pass all rules and have unique order IDs. The dashboard recalculates these counts from the uploaded file; they may differ if the Kaggle file changes. This relatively low retention rate should be reported and discussed rather than hidden.

## 2. What the experiment actually measures

1. The app checks that the uploaded file contains the 16 expected columns.
2. Predeclared benchmark eligibility rules exclude rows with missing required values, out-of-range values under the study's stated inclusion rules, blank categories, or duplicate order IDs. These are **our benchmark inclusion rules**, not rules attributed to Kaggle.
3. A reproducible sample is selected using a recorded random seed. Its canonicalized values are preserved as the target ground truth.
4. Controlled, reversible formatting and type variations are applied to those records (for example, numeric values with units, percentage strings, title/upper case, extra whitespace, readable flag labels and a prefixed order ID).
5. Both pipelines receive the **same corrupted records**. The rule-based system uses explicit Python rules; the LLM system sends records to the configured Groq model and expects structured JSON.
6. Outputs are compared field by field and record by record against the preserved ground truth.
7. The app records run metadata and lets the researcher export an audit JSON and record-level CSV.

### Metrics

- **Field-level accuracy:** share of target fields restored correctly, using declared numeric tolerances.
- **Complete-record accuracy:** share of records for which every target field is correct.
- **Observed processing time and throughput:** timing for the actual selected sample only.
- **Token usage and service-equivalent cost estimate:** calculated from API-reported token counts and the editable reference prices in the sidebar. This is not a billing statement; actual charges depend on the applicable provider plan and allowance.
- **Failures:** API exceptions and incomplete or invalid model outputs remain visible and are not silently treated as correct.

Only tested sample sizes (20, 50 and/or 100) are shown. The app does **not** extrapolate measured runtime or claim that a million-record dataset was tested. Rule-based local execution time and LLM API wall time have different measurement conditions and must be discussed honestly.

## 3. Limitations and research integrity

- This is a bounded, controlled formatting/type-normalization experiment. It does not evaluate semantic repair of unknown errors, missing-value imputation, schema mapping across unrelated systems, or production ETL reliability.
- Because the corruption patterns are known and reversible, a correctly implemented rule-based baseline is expected to perform strongly. The study should test when an LLM justifies its added complexity, cost, or latency—not presume the LLM will win.
- The dataset provenance has not been independently established. Do not claim that the results generalize to all small businesses or represent observed outcomes at a verified retailer.
- The cost estimate uses the rates shown in the code/sidebar and should be checked against current Groq model pricing before reporting. A free allowance may make the actual bill $0 while a service-equivalent estimate remains positive.
- Preserve the exact model ID, experiment date, Git commit, seed, selected sample size, audit JSON, record-level CSV and any errors. Repeat the experiment with predeclared seeds if time and free-tier limits allow.
- Do not transfer the old ICICV manuscript's numerical results to this new benchmark. Results must be recalculated from these new runs.

## 4. Configure Groq API access

1. Sign in to [Groq Console](https://console.groq.com/).
2. Open **API Keys** and create a key.
3. In Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "paste-your-private-key-here"
GROQ_MODEL = "openai/gpt-oss-20b"
```

The `GROQ_MODEL` setting is optional; the app defaults to `openai/gpt-oss-20b`. Verify that the configured model is currently available to your account. Never place the key in `app.py`, this README, a public GitHub commit, a screenshot, or an audit export.

Groq documentation: [supported models and prices](https://console.groq.com/docs/models) · [rate limits](https://console.groq.com/docs/rate-limits).

The dashboard's default reference rates are $0.075 per million input tokens and $0.30 per million output tokens for service-equivalent cost estimation. The sidebar permits adjusting them. Verify current provider documentation when running the experiment; do not treat these defaults as guaranteed future prices.

## 5. Deploy or update the Streamlit app

### Files in this project

- `app.py` — research dashboard, dataset checks, controlled benchmark generation, both pipelines, charts and exports.
- `requirements.txt` — Python dependencies used by Streamlit Cloud.
- `.gitignore` — prevents common local credentials, datasets and experiment outputs from being committed.
- `.streamlit/config.toml` — optional dark theme and gold accent consistent with the dashboard.
- `tests/test_benchmark.py` — smoke tests for the deterministic ETL and evaluation logic.

### Update the existing `LLM-Experiment-Pipeline` repository

Repository: <https://github.com/Anish0605/LLM-Experiment-Pipeline>

1. **Make a backup first.** Download the current repository ZIP or create a branch named `backup-before-icitiit27` so the earlier NGNDAI 2026 app remains recoverable.
2. In the repository, upload the replacement `app.py`, `README.md`, `requirements.txt` and `.gitignore` from this package to the repository root using **Add file → Upload files**.
3. Commit the changes. If you upload to `main`, the existing Streamlit deployment may rebuild from that branch. If you use a new branch, select that branch in the Streamlit app settings or create a separate Streamlit deployment.
4. Open the Streamlit app's **Manage app → Settings → Secrets** and add the Groq settings shown above.
5. Wait for the app to rebuild, then use **Test Groq connection** in the sidebar.
6. Upload `train.csv`, build the benchmark, run the rule-based baseline, run 20 LLM records first, review outputs, and then extend to 50 and 100 if the free quota permits.

### Local run (optional)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
export GROQ_API_KEY="your-private-key"
streamlit run app.py
```

On Windows, activate the virtual environment using `.venv\\Scripts\\activate` and set the environment variable using the appropriate PowerShell command. Do not commit environment files.

## 6. Smoke tests

With dependencies installed, run the deterministic logic tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests check that the rule-based transform restores the controlled formatting variants, missing model fields count as incorrect, and the canonical schema contains the expected fields. They do not test the live Groq service or establish empirical paper results.

## 7. Experiment workflow

1. Download the data using the Kaggle competition page. Upload `train.csv` in the app (not `test.csv` or `sample_submission.csv`).
2. Inspect source row counts and the individual eligibility rules.
3. Build a reproducible 20, 50 or 100 record sample. The app records a SHA-256 digest of the uploaded file and the random seed.
4. Run the rule-based baseline.
5. Test the Groq connection and run the LLM pipeline for 20 records.
6. Review field accuracy and record accuracy, inspect errors, and download the JSON audit trail and CSV output.
7. Extend to 50 and 100 only after the small pilot succeeds and account limits permit. Download the audit files before leaving the session.
8. Do not interpret a valid JSON response as a correct transformation: only agreement with the stored target ground truth receives credit.

## 8. ICITIIT 2027 submission reference

Conference author instructions: <https://icitiit27.iiitkottayam.ac.in/instructions-to-authors/>.

The official site has listed a 20 October 2026 submission deadline, a six-page review-paper limit including references, and an IEEE two-column format. These details may change, so confirm the current official instructions before submission. The repository is a reproducibility aid; it does not establish that the paper's experimental results are complete or that the paper will be accepted.
