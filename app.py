"""ICITIIT 2027 controlled ETL benchmark dashboard.

Compares deterministic rule-based normalization with Groq-hosted LLM normalization on
identical, controlled formatting variations. This app does not train a return predictor.
The pre-corruption canonical records are the ground truth for the controlled task.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import uuid
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="ICITIIT 2027 · ETL Benchmark",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Current public Groq developer-plan reference rates when this package was prepared.
# They are an estimate of service-equivalent cost, NOT necessarily the amount billed.
DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_INPUT_USD_PER_MILLION = 0.075
DEFAULT_OUTPUT_USD_PER_MILLION = 0.30
DATA_SOURCE_URL = "https://www.kaggle.com/competitions/retail-return-risk-modeling"
DATASET_TITLE = "E-Commerce Returns: Who Will Send It Back?"
APP_VERSION = "1.0.0"

FIELDS = [
    "customer_age", "product_price", "discount_percent", "product_rating",
    "past_purchase_count", "past_return_rate", "delivery_delay_days",
    "session_length_minutes", "num_product_views", "device_type",
    "product_category", "shipping_method", "payment_method", "used_coupon",
    "returned", "order_id",
]
NUMERIC_FIELDS = {
    "customer_age", "product_price", "discount_percent", "product_rating",
    "past_purchase_count", "past_return_rate", "delivery_delay_days",
    "session_length_minutes", "num_product_views", "used_coupon", "returned",
}
INTEGER_FIELDS = {"customer_age", "past_purchase_count", "num_product_views", "used_coupon", "returned"}
STRING_FIELDS = {"device_type", "product_category", "shipping_method", "payment_method"}
DECIMAL_2_FIELDS = {"product_price", "discount_percent", "product_rating", "delivery_delay_days", "session_length_minutes"}

CSS = r"""
<style>
:root { --ink:#243247; --muted:#5f6b7a; --gold:#eab330; --green:#276749; --line:#dbe2ea; }
.block-container { max-width:1480px; padding-top:1.25rem; padding-bottom:2.6rem; }
[data-testid="stAppViewContainer"] { background:#f7f8fa; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(247,248,250,.96); }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#243247; }
.hero-panel { padding:1.45rem 1.7rem; margin-bottom:1.15rem; border-radius:12px; border:1px solid var(--line); border-left:5px solid var(--gold); background:#ffffff; box-shadow:0 2px 8px rgba(36,50,71,.04); }
.hero-kicker { color:#8a6500; font-weight:750; letter-spacing:.09em; font-size:.76rem; text-transform:uppercase; margin-bottom:.55rem; }
.hero-panel h1 { color:#172b4d; font-size:clamp(1.6rem,3vw,2.25rem); line-height:1.18; margin:0 0 .55rem 0; }
.hero-subtitle { color:#334155; font-size:1.04rem; margin-bottom:.45rem; }
.hero-detail { color:#5f6b7a; font-size:.86rem; margin-top:.7rem; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--line); border-top:3px solid var(--gold); border-radius:10px; padding:.9rem 1rem; min-height:112px; box-shadow:0 1px 4px rgba(36,50,71,.03); }
[data-testid="stMetricLabel"] p { color:#596579 !important; font-size:.84rem !important; }
[data-testid="stMetricValue"] { color:#172b4d !important; }
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:7px 7px 0 0; padding:.6rem .8rem; }
.stTabs [aria-selected="true"] { color:#1f6f50 !important; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:9px; background:#ffffff; }
div.stButton > button[kind="primary"] { background:#276749; color:#ffffff; border:1px solid #276749; font-weight:650; }
div.stButton > button[kind="primary"]:hover { background:#1f553c; border-color:#1f553c; color:#ffffff; }
.small-muted { color:#5f6b7a; font-size:.84rem; }
.section-label { color:#8a6500; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; font-weight:750; }
.status-card { background:#ffffff; border:1px solid var(--line); border-radius:9px; padding:.8rem 1rem; }
</style>
"""


def get_setting(name: str, default: str = "") -> str:
    """Read a value from Streamlit Secrets first, then environment variables."""
    try:
        val = st.secrets.get(name, "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip() or default


def get_api_key() -> str:
    return get_setting("GROQ_API_KEY", "")


def kaggle_credentials_available() -> bool:
    """Return True if Kaggle competition API credentials are configured."""
    username = get_setting("KAGGLE_USERNAME", "")
    key = get_setting("KAGGLE_KEY", "")
    return bool(username and key)


def download_kaggle_train_csv() -> tuple[bytes, str]:
    """Download train.csv from the Kaggle competition using configured secrets.

    The user must accept the competition rules on Kaggle before the API can access
    competition files. Credentials are read from Streamlit Secrets/environment only.
    """
    username = get_setting("KAGGLE_USERNAME", "")
    key = get_setting("KAGGLE_KEY", "")
    if not username or not key:
        raise RuntimeError(
            "Kaggle API credentials are missing. Add KAGGLE_USERNAME and KAGGLE_KEY "
            "to Streamlit App Settings → Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    with tempfile.TemporaryDirectory(prefix="icitiit_kaggle_") as tmp:
        command = [
            "kaggle", "competitions", "download",
            "-c", "retail-return-risk-modeling", "-f", "train.csv", "-p", tmp,
        ]
        try:
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=180, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Kaggle download timed out after 180 seconds. Please retry.") from exc

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "No details returned")[-1800:]
            if "403" in details or "rules" in details.lower() or "accept" in details.lower():
                raise RuntimeError(
                    "Kaggle did not allow the download. Open the competition page while signed in, "
                    "accept its rules/terms, then retry. API details: " + details
                )
            raise RuntimeError(
                "Kaggle download failed. Check that KAGGLE_USERNAME/KAGGLE_KEY are correct "
                "and the competition is accessible to your account. Details: " + details
            )

        candidates = list(Path(tmp).glob("*.zip"))
        direct_csv = Path(tmp) / "train.csv"
        if direct_csv.exists():
            data = direct_csv.read_bytes()
        elif candidates:
            with zipfile.ZipFile(candidates[0]) as archive:
                names = [n for n in archive.namelist() if Path(n).name.lower() == "train.csv"]
                if not names:
                    raise RuntimeError(
                        "Kaggle download succeeded but the archive did not contain train.csv. "
                        "Files found: " + ", ".join(archive.namelist()[:20])
                    )
                data = archive.read(names[0])
        else:
            raise RuntimeError("Kaggle command completed but no train.csv or ZIP archive was found.")
        return data, "Kaggle API · retail-return-risk-modeling/train.csv"


def init_state() -> None:
    defaults: dict[str, Any] = {
        "source_fingerprint": None,
        "source_filename": None,
        "source_row_count": 0,
        "source_eligible_count": 0,
        "source_columns": [],
        "source_loaded_at": None,
        "gold_records": {},
        "corrupted_records": {},
        "rule_outputs": {},
        "rule_times": {},
        "llm_outputs": {},
        "llm_meta_by_key": {},
        "llm_calls": [],
        "seed": 42,
        "benchmark_size": 0,
        "benchmark_built_at": None,
        "benchmark_fingerprint": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_benchmark() -> None:
    for key, value in {
        "gold_records": {}, "corrupted_records": {}, "rule_outputs": {}, "rule_times": {},
        "llm_outputs": {}, "llm_meta_by_key": {}, "llm_calls": [], "seed": 42,
        "benchmark_size": 0, "benchmark_built_at": None, "benchmark_fingerprint": None,
    }.items():
        st.session_state[key] = value


@st.cache_data(show_spinner=False, max_entries=3)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def eligibility_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.Series]:
    """Apply documented benchmark inclusion rules; return working data and rule counts."""
    missing_cols = [c for c in FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing_cols))

    d = df[FIELDS].copy()
    for col in NUMERIC_FIELDS:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    rules: dict[str, pd.Series] = {
        "All 16 required fields present": d[FIELDS].notna().all(axis=1),
        "Customer age between 18 and 100": d["customer_age"].between(18, 100),
        "Product price greater than zero": d["product_price"].gt(0),
        "Discount between 0 and 100": d["discount_percent"].between(0, 100),
        "Product rating between 1 and 5": d["product_rating"].between(1, 5),
        "Purchase count non-negative": d["past_purchase_count"].ge(0),
        "Return rate between 0 and 1": d["past_return_rate"].between(0, 1),
        "Delivery delay non-negative": d["delivery_delay_days"].ge(0),
        "Session length non-negative": d["session_length_minutes"].ge(0),
        "Product views non-negative": d["num_product_views"].ge(0),
        "Coupon flag is binary": d["used_coupon"].isin([0, 1]),
        "Return flag is binary": d["returned"].isin([0, 1]),
    }
    for col in STRING_FIELDS:
        rules[f"{col} is non-empty"] = d[col].astype("string").str.strip().ne("").fillna(False)
    rules["Order ID is unique"] = ~d["order_id"].astype("string").str.strip().duplicated(keep="first")

    combined = pd.Series(True, index=d.index)
    for result in rules.values():
        combined &= result.fillna(False)
    counts = {name: int(result.fillna(False).sum()) for name, result in rules.items()}
    eligible = d.loc[combined].copy()
    return eligible, counts, combined


def numeric_value(value: Any, percent_to_fraction: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower().replace(",", "")
    has_percent = "%" in text
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None
    try:
        number = float(match.group(0))
        if has_percent and percent_to_fraction:
            number /= 100.0
        return number
    except ValueError:
        return None


def normalize_order_id(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip()


def canonicalize(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Declare canonical field types/precision before the corruption is introduced."""
    d = dict(row)
    out: dict[str, Any] = {
        "customer_age": int(round(float(d["customer_age"]))),
        "product_price": round(float(d["product_price"]), 2),
        "discount_percent": round(float(d["discount_percent"]), 2),
        "product_rating": round(float(d["product_rating"]), 2),
        "past_purchase_count": int(round(float(d["past_purchase_count"]))),
        "past_return_rate": round(float(d["past_return_rate"]), 4),
        "delivery_delay_days": round(float(d["delivery_delay_days"]), 2),
        "session_length_minutes": round(float(d["session_length_minutes"]), 2),
        "num_product_views": int(round(float(d["num_product_views"]))),
        "used_coupon": int(round(float(d["used_coupon"]))),
        "returned": int(round(float(d["returned"]))),
        "order_id": normalize_order_id(d["order_id"]),
    }
    for field in STRING_FIELDS:
        out[field] = str(d[field]).strip().lower()
    return {field: out[field] for field in FIELDS}


def corrupt_record(gold: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Apply controlled, reversible formatting/type variations; do not alter the value."""
    item = dict(gold)
    if row_index % 2 == 0:
        item["customer_age"] = f" {gold['customer_age']} years "
        item["product_price"] = f"price {gold['product_price']:.2f}"
        item["discount_percent"] = f"{gold['discount_percent']:.2f} %"
        item["product_rating"] = f"{gold['product_rating']:.2f} stars"
        item["past_purchase_count"] = f"{gold['past_purchase_count']} purchases"
        item["past_return_rate"] = f"{gold['past_return_rate'] * 100:.2f}%"
        item["delivery_delay_days"] = f"{gold['delivery_delay_days']:.2f} days"
        item["session_length_minutes"] = f"{gold['session_length_minutes']:.2f} min"
        item["num_product_views"] = f"{gold['num_product_views']} views"
    else:
        item["customer_age"] = str(gold["customer_age"])
        item["product_price"] = f"{gold['product_price']:.2f} currency-units"
        item["discount_percent"] = f"{gold['discount_percent']:.2f} percent"
        item["product_rating"] = f"rating: {gold['product_rating']:.2f}/5"
        item["past_purchase_count"] = f"count={gold['past_purchase_count']}"
        item["past_return_rate"] = f"rate={gold['past_return_rate']:.4f}"
        item["delivery_delay_days"] = f"delay {gold['delivery_delay_days']:.2f}"
        item["session_length_minutes"] = f"{gold['session_length_minutes']:.2f} minutes"
        item["num_product_views"] = f"views={gold['num_product_views']}"
    for field in STRING_FIELDS:
        item[field] = f"  {str(gold[field]).upper()}  " if row_index % 2 == 0 else f"{str(gold[field]).title()} "
    item["used_coupon"] = " YES " if gold["used_coupon"] == 1 else "no"
    item["returned"] = "Returned" if gold["returned"] == 1 else "Not returned"
    item["order_id"] = f"ORD-{gold['order_id']}"
    return item


def parse_numeric(value: Any, field: str) -> Any:
    value_num = numeric_value(value, percent_to_fraction=(field == "past_return_rate"))
    if value_num is None:
        return value
    if field in INTEGER_FIELDS:
        return int(round(value_num))
    if field in DECIMAL_2_FIELDS:
        return round(value_num, 2)
    if field == "past_return_rate":
        return round(value_num, 4)
    return value_num


def rule_transform(corrupted: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        value = corrupted.get(field)
        if field == "used_coupon":
            low = str(value).strip().lower()
            out[field] = 1 if low in {"yes", "true", "1"} else 0 if low in {"no", "false", "0"} else value
        elif field == "returned":
            low = str(value).strip().lower()
            out[field] = 1 if low in {"returned", "yes", "true", "1"} else 0 if low in {"not returned", "no", "false", "0"} else value
        else:
            out[field] = parse_numeric(value, field)
    for field in STRING_FIELDS:
        out[field] = str(corrupted.get(field, "")).strip().lower()
    order_id = str(corrupted.get("order_id", "")).strip()
    out["order_id"] = re.sub(r"^ord-", "", order_id, flags=re.IGNORECASE).strip()
    return {field: out.get(field) for field in FIELDS}


def values_equal(field: str, predicted: Any, target: Any) -> bool:
    if predicted is None:
        return False
    if field in NUMERIC_FIELDS:
        p = numeric_value(predicted, percent_to_fraction=(field == "past_return_rate"))
        g = numeric_value(target, percent_to_fraction=(field == "past_return_rate"))
        if p is None or g is None:
            return False
        tolerance = 1e-6 if field in INTEGER_FIELDS or field == "past_return_rate" else 0.001
        return abs(p - g) <= tolerance
    p_text = str(predicted).strip().lower()
    g_text = str(target).strip().lower()
    if field == "order_id":
        p_text = re.sub(r"^ord-", "", p_text, flags=re.IGNORECASE)
        g_text = re.sub(r"^ord-", "", g_text, flags=re.IGNORECASE)
    return p_text == g_text


def evaluate_outputs(
    gold_records: dict[str, dict[str, Any]],
    predicted_records: dict[str, dict[str, Any]],
    keys: list[str],
    method: str,
) -> dict[str, Any]:
    field_total = len(keys) * len(FIELDS)
    field_correct = 0
    fully_correct = 0
    for key in keys:
        gold = gold_records[key]
        candidate = predicted_records.get(key, {})
        candidate = candidate if isinstance(candidate, dict) else {}
        matched = [values_equal(field, candidate.get(field), gold.get(field)) for field in FIELDS]
        field_correct += sum(matched)
        fully_correct += int(bool(matched) and all(matched))
    return {
        "method": method,
        "records": len(keys),
        "field_accuracy_pct": round(100 * field_correct / field_total, 2) if field_total else 0.0,
        "record_accuracy_pct": round(100 * fully_correct / len(keys), 2) if keys else 0.0,
        "correct_fields": field_correct,
        "total_fields": field_total,
        "fully_correct_records": fully_correct,
    }


def model_prompt(batch_rows: list[dict[str, Any]]) -> str:
    return f"""Normalize records for a controlled ETL benchmark. Each row was generated from a known canonical record using reversible formatting and type changes. Restore canonical types and values; do not predict outcomes or invent values.

Target rules:
- customer_age: integer years
- product_price: numeric value rounded to 2 decimal places; source currency/unit is not verified
- discount_percent: numeric percentage from 0 to 100, not a fraction
- product_rating: numeric rating from 1 to 5, rounded to 2 decimals
- past_purchase_count: integer
- past_return_rate: decimal fraction from 0 to 1 (e.g., 0.25 means 25%), rounded to 4 decimals
- delivery_delay_days: numeric days rounded to 2 decimals
- session_length_minutes: numeric minutes rounded to 2 decimals
- num_product_views: integer
- device_type, product_category, shipping_method, payment_method: lowercase strings, trimmed
- used_coupon: integer 0 or 1
- returned: integer 0 or 1 (Returned=1; Not returned=0)
- order_id: string with optional leading ORD- prefix removed

Return one JSON object with key \"records\" containing an array. Each row must contain \"record_key\" and exactly the 16 schema fields above. Preserve the record_key exactly. No markdown or explanation. Numeric fields must be JSON numbers, without units or currency symbols.

INPUT RECORDS:
{json.dumps(batch_rows, ensure_ascii=False, separators=(',', ':'))}
"""


def call_groq(
    client: Groq,
    batch_rows: list[dict[str, Any]],
    model_id: str,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    raw_text = ""
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You normalize data records to an exact schema. Return valid JSON only."},
            {"role": "user", "content": model_prompt(batch_rows)},
        ],
        temperature=0,
        max_tokens=3500,
        response_format={"type": "json_object"},
    )
    elapsed = time.perf_counter() - started
    raw_text = completion.choices[0].message.content or "{}"
    parsed = json.loads(raw_text)
    rows = parsed.get("records", []) if isinstance(parsed, dict) else parsed
    output: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("record_key") is None:
            continue
        key = str(row["record_key"])
        output[key] = {field: row.get(field) for field in FIELDS}
    usage = getattr(completion, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    has_usage = (
        usage is not None
        and getattr(usage, "prompt_tokens", None) is not None
        and getattr(usage, "completion_tokens", None) is not None
    )
    estimated_cost = (
        (input_tokens * input_usd_per_million + output_tokens * output_usd_per_million) / 1_000_000
        if has_usage else None
    )
    meta = {
        "call_id": f"call_{uuid.uuid4().hex[:12]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "record_keys": [str(r.get("record_key", "")) for r in batch_rows],
        "elapsed_sec": round(elapsed, 5),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "has_usage": has_usage,
        "estimated_service_cost_usd": round(estimated_cost, 8) if estimated_cost is not None else None,
        "batch_records": len(batch_rows),
        "error": None,
        "raw_response": raw_text,
    }
    return output, meta


def failed_call_meta(batch_rows: list[dict[str, Any]], model_id: str, elapsed: float, exc: Exception) -> dict[str, Any]:
    return {
        "call_id": f"call_{uuid.uuid4().hex[:12]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "record_keys": [str(r.get("record_key", "")) for r in batch_rows],
        "elapsed_sec": round(elapsed, 5),
        "input_tokens": 0,
        "output_tokens": 0,
        "has_usage": False,
        "estimated_service_cost_usd": None,
        "batch_records": len(batch_rows),
        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        "raw_response": None,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_result_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summaries only for sample sizes actually completed, never extrapolated."""
    gold: dict[str, dict[str, Any]] = st.session_state.gold_records
    rule: dict[str, dict[str, Any]] = st.session_state.rule_outputs
    llm: dict[str, dict[str, Any]] = st.session_state.llm_outputs
    calls: list[dict[str, Any]] = st.session_state.llm_calls
    keys = list(gold)
    summaries: list[dict[str, Any]] = []
    for n in (20, 50, 100):
        if n > len(keys) or not all(key in llm for key in keys[:n]):
            continue
        sample_keys = keys[:n]
        rule_eval = evaluate_outputs(gold, rule, sample_keys, "Rule-based")
        rule_seconds = sum(float(st.session_state.rule_times.get(key, 0.0)) for key in sample_keys)
        summaries.append({
            **rule_eval,
            "sample_size": n,
            "processing_seconds": rule_seconds,
            "records_per_second": n / rule_seconds if rule_seconds > 0 else None,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_service_cost_usd": 0.0,
            "attempted_api_calls": 0,
            "api_errors": 0,
        })

        prefix = set(sample_keys)
        relevant_calls = [call for call in calls if call.get("record_keys") and set(call["record_keys"]).issubset(prefix)]
        llm_eval = evaluate_outputs(gold, llm, sample_keys, "LLM-assisted")
        llm_seconds = sum(float(call.get("elapsed_sec") or 0) for call in relevant_calls)
        token_counts_available = all(call.get("has_usage", False) for call in relevant_calls) if relevant_calls else False
        total_cost = (
            sum(float(call.get("estimated_service_cost_usd") or 0) for call in relevant_calls)
            if token_counts_available else None
        )
        summaries.append({
            **llm_eval,
            "sample_size": n,
            "processing_seconds": llm_seconds,
            "records_per_second": n / llm_seconds if llm_seconds > 0 else None,
            "input_tokens": sum(int(call.get("input_tokens") or 0) for call in relevant_calls),
            "output_tokens": sum(int(call.get("output_tokens") or 0) for call in relevant_calls),
            "estimated_service_cost_usd": total_cost,
            "attempted_api_calls": len(relevant_calls),
            "api_errors": sum(1 for call in relevant_calls if call.get("error")),
        })
    summary_df = pd.DataFrame(summaries)

    field_rows: list[dict[str, Any]] = []
    if not summary_df.empty:
        sizes = [int(n) for n in summary_df["sample_size"].unique()]
        if sizes:
            largest_n = max(sizes)
            largest_keys = keys[:largest_n]
            for field in FIELDS:
                for method, outputs in (("Rule-based", rule), ("LLM-assisted", llm)):
                    correct = sum(values_equal(field, outputs.get(k, {}).get(field), gold[k][field]) for k in largest_keys)
                    field_rows.append({
                        "field": field,
                        "method": method,
                        "correct": int(correct),
                        "records": largest_n,
                        "accuracy_pct": 100 * correct / largest_n if largest_n else 0,
                    })
    return summary_df, pd.DataFrame(field_rows)


def make_audit_payload() -> dict[str, Any]:
    return {
        "metadata": {
            "application": "ICITIIT 2027 Controlled ETL Benchmark",
            "app_version": APP_VERSION,
            "research_title": "AI-Augmented Marketing Analytics: Evaluating Large Language Models in ETL Pipelines for Accuracy, Cost, and Scalability in Small-Scale Deployments",
            "dataset_title": DATASET_TITLE,
            "dataset_source_url": DATA_SOURCE_URL,
            "uploaded_filename": st.session_state.source_filename,
            "uploaded_file_sha256": st.session_state.source_fingerprint,
            "uploaded_rows": st.session_state.source_row_count,
            "eligible_rows": st.session_state.source_eligible_count,
            "model_id": get_setting("GROQ_MODEL", DEFAULT_MODEL_ID),
            "random_seed": st.session_state.seed,
            "benchmark_size": st.session_state.benchmark_size,
            "benchmark_built_at_utc": st.session_state.benchmark_built_at,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_fields": FIELDS,
            "input_usd_per_million_tokens_reference": float(st.session_state.get("input_rate", DEFAULT_INPUT_USD_PER_MILLION)),
            "output_usd_per_million_tokens_reference": float(st.session_state.get("output_rate", DEFAULT_OUTPUT_USD_PER_MILLION)),
            "method_note": "The ground truth is canonicalized eligible source rows before reversible formatting corruption. The benchmark covers controlled format/type normalization, not semantic repair or return prediction.",
            "provenance_note": "The competition listing describes records from 2022-2024; originating retailer and real-world provenance have not been independently verified.",
        },
        "gold_records": st.session_state.gold_records,
        "corrupted_records": st.session_state.corrupted_records,
        "rule_outputs": st.session_state.rule_outputs,
        "rule_times_seconds": st.session_state.rule_times,
        "llm_outputs": st.session_state.llm_outputs,
        "llm_meta_by_key": st.session_state.llm_meta_by_key,
        "llm_calls": st.session_state.llm_calls,
    }


def style_plot(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5f4", family="Inter, Arial, sans-serif"),
        margin=dict(l=12, r=12, t=56, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(font=dict(size=16)),
    )
    fig.update_xaxes(gridcolor="#2a3850", zerolinecolor="#2a3850")
    fig.update_yaxes(gridcolor="#2a3850", zerolinecolor="#2a3850")
    return fig


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    api_key = get_api_key()
    model_id = get_setting("GROQ_MODEL", DEFAULT_MODEL_ID)

    st.markdown(
        """<div class="hero-panel">
        <div class="hero-kicker">ICITIIT 2027 · RESEARCH EXPERIMENT CONSOLE</div>
        <h1>Controlled ETL Benchmark</h1>
        <div class="hero-subtitle">Rule-based ETL vs. LLM-assisted ETL</div>
        <div>Measure transformation accuracy, observed processing time, token usage and service-equivalent cost on the same controlled records.</div>
        <div class="hero-detail">AI-Augmented Marketing Analytics: Evaluating Large Language Models in ETL Pipelines for Accuracy, Cost, and Scalability in Small-Scale Deployments</div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🔐 Model & API")
        st.caption(f"Model ID: `{model_id}`")
        if api_key:
            st.success("GROQ_API_KEY detected in secure configuration")
        else:
            st.error("GROQ_API_KEY not configured")
            st.caption("Add it under Streamlit App → Settings → Secrets. Never put the key in GitHub code.")
        if st.button("Test Groq connection", disabled=not bool(api_key), use_container_width=True):
            try:
                with st.spinner("Checking Groq API access…"):
                    model_list = Groq(api_key=api_key).models.list()
                    available_ids = [getattr(m, "id", "") for m in getattr(model_list, "data", [])]
                if model_id in available_ids:
                    st.success(f"Connection OK; model `{model_id}` is listed.")
                else:
                    st.warning("API connection succeeded, but the configured model was not in the returned model list. Check model access and exact ID before running.")
            except Exception as exc:
                st.error(f"Connection check failed: {type(exc).__name__}: {str(exc)[:250]}")

        st.markdown("---")
        st.markdown("### 💵 Cost estimate settings")
        input_rate = st.number_input("Input USD / 1M tokens", min_value=0.0, value=DEFAULT_INPUT_USD_PER_MILLION, step=0.025, format="%.4f")
        output_rate = st.number_input("Output USD / 1M tokens", min_value=0.0, value=DEFAULT_OUTPUT_USD_PER_MILLION, step=0.05, format="%.4f")
        st.session_state.input_rate = float(input_rate)
        st.session_state.output_rate = float(output_rate)
        st.caption("Reference rates are editable estimates, not a billing statement. Verify current provider pricing before publication.")

        st.markdown("---")
        st.markdown("### 📋 Experiment progress")
        progress_items = [
            ("1 · Training CSV loaded", bool(st.session_state.source_fingerprint)),
            ("2 · Benchmark built", bool(st.session_state.gold_records)),
            ("3 · Rule baseline run", bool(st.session_state.rule_outputs)),
            ("4 · LLM outputs collected", bool(st.session_state.llm_outputs)),
        ]
        for label, done in progress_items:
            st.markdown(("✅" if done else "⬜") + " " + label)
        if st.session_state.gold_records:
            st.caption(f"Benchmark pool: {st.session_state.benchmark_size} records · seed {st.session_state.seed}")

        st.markdown("---")
        st.markdown("### 📚 Source")
        st.markdown(f"[{DATASET_TITLE}]({DATA_SOURCE_URL})")
        st.caption("Upload `train.csv` only. Test and sample-submission files are not ETL ground truth.")

    st.markdown("<div class='section-label'>Dataset workspace</div>", unsafe_allow_html=True)
    source_col, upload_col = st.columns([1.05, 1])
    with source_col:
        st.markdown(
            f"<div class='status-card'><b>Dataset source</b><br><a href='{DATA_SOURCE_URL}' target='_blank'>{DATASET_TITLE}</a><br><span class='small-muted'>The app retrieves train.csv directly through the Kaggle API. Competition access requires accepting the rules on Kaggle. The records' real-world origin is not independently verified.</span></div>",
            unsafe_allow_html=True,
        )
        kaggle_ready = kaggle_credentials_available()
        if kaggle_ready:
            st.success("Kaggle API credentials configured in Streamlit Secrets.")
        else:
            st.info("For one-click download, add KAGGLE_USERNAME and KAGGLE_KEY in Streamlit Secrets.")
        if st.button("Download train.csv from Kaggle", type="primary", disabled=not kaggle_ready, key="download_kaggle_train"):
            with st.spinner("Connecting to Kaggle and downloading train.csv…"):
                try:
                    kaggle_bytes, kaggle_label = download_kaggle_train_csv()
                    st.session_state["kaggle_train_bytes"] = kaggle_bytes
                    st.session_state["kaggle_train_label"] = kaggle_label
                    st.success(f"Downloaded train.csv ({len(kaggle_bytes):,} bytes).")
                except Exception as exc:
                    st.error(f"Kaggle download failed: {exc}")
    with upload_col:
        uploaded = st.file_uploader(
            "Optional fallback: upload `train.csv` manually",
            type=["csv"], key="train_csv",
            help="Use this only if direct Kaggle API access is unavailable. Do not commit the CSV to GitHub.",
        )
        if st.session_state.get("kaggle_train_bytes"):
            st.caption("Dataset loaded from Kaggle API. A manual upload, if provided, takes priority.")

    source_df: pd.DataFrame | None = None
    eligible_df: pd.DataFrame | None = None
    eligibility_counts: dict[str, int] = {}
    eligibility_mask: pd.Series | None = None
    file_bytes: bytes | None = None
    schema_error: str | None = None

    selected_filename = None
    if uploaded is not None or st.session_state.get("kaggle_train_bytes"):
        try:
            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                selected_filename = uploaded.name
            else:
                file_bytes = st.session_state["kaggle_train_bytes"]
                selected_filename = "train.csv (Kaggle API)"
            fingerprint = hashlib.sha256(file_bytes).hexdigest()
            if st.session_state.source_fingerprint and st.session_state.source_fingerprint != fingerprint:
                reset_benchmark()
                st.toast("Dataset changed; the previous benchmark run was cleared to prevent mixing datasets.")
            source_df = read_csv_bytes(file_bytes)
            missing_fields = [field for field in FIELDS if field not in source_df.columns]
            if missing_fields:
                schema_error = "This CSV does not contain the required competition columns: " + ", ".join(missing_fields)
            else:
                eligible_df, eligibility_counts, eligibility_mask = eligibility_rules(source_df)
                st.session_state.source_fingerprint = fingerprint
                st.session_state.source_filename = selected_filename
                st.session_state.source_row_count = int(len(source_df))
                st.session_state.source_eligible_count = int(len(eligible_df))
                st.session_state.source_columns = list(source_df.columns)
                st.session_state.source_loaded_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            schema_error = f"Could not read or validate the uploaded CSV: {type(exc).__name__}: {str(exc)[:300]}"
    elif st.session_state.source_fingerprint:
        reset_benchmark()
        for key in ("source_fingerprint", "source_filename", "source_row_count", "source_eligible_count", "source_columns", "source_loaded_at"):
            st.session_state[key] = None if key in {"source_fingerprint", "source_filename", "source_loaded_at"} else (0 if key.endswith("count") else [])

    if schema_error:
        st.error(schema_error)

    tab_overview, tab_dataset, tab_run, tab_results, tab_audit, tab_protocol = st.tabs(
        ["📊 Overview", "🗂️ Dataset & benchmark", "⚡ Run pipelines", "📈 Results", "📦 Audit & export", "🧭 Research protocol"]
    )

    with tab_overview:
        st.subheader("Experiment at a glance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source rows", f"{len(source_df):,}" if source_df is not None else "—")
        c2.metric("Eligible rows", f"{len(eligible_df):,}" if eligible_df is not None else "—")
        c3.metric("Benchmark size", f"{st.session_state.benchmark_size:,}" if st.session_state.benchmark_size else "Not built")
        c4.metric("LLM records processed", f"{len(st.session_state.llm_outputs):,}")
        st.markdown("#### Research question")
        st.write("For a fixed, known set of e-commerce records with reversible formatting/type variations, how do a deterministic Python/Pandas pipeline and an LLM-assisted pipeline compare on field accuracy, complete-record accuracy, measured elapsed time, throughput and token-based service-equivalent cost?")
        st.info("This application evaluates data transformation only. It does not train a return-prediction model, and the `returned` column is treated as one ordinary field to normalize.")
        if source_df is None:
            st.warning("Start by uploading the Kaggle competition `train.csv` above. The dashboard will then validate the schema and enable the benchmark workflow.")
        else:
            if not schema_error:
                pct = 100 * len(eligible_df) / len(source_df) if len(source_df) else 0
                st.markdown("#### Source eligibility")
                st.progress(min(max(pct / 100, 0), 1))
                st.caption(f"{len(eligible_df):,} of {len(source_df):,} source records meet the declared inclusion rules ({pct:.1f}%). Eligibility rules are benchmark choices; they are not asserted to be the dataset creator's own quality rules.")
        st.markdown("#### Research integrity note")
        st.caption("All reported outcomes must come from the run exports. Do not reuse the previous manuscript's 94.1% / 72.4% figures as results for this new experiment. The competition's listed period is 2022–2024; the original retailer and real-world provenance have not been independently verified.")

    with tab_dataset:
        st.subheader("1 · Validate and inspect the dataset")
        if source_df is None or schema_error or eligible_df is None:
            st.info("Upload a schema-matching `train.csv` file to see the dataset profile and build a controlled benchmark.")
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Uploaded rows", f"{len(source_df):,}")
            d2.metric("Rows eligible", f"{len(eligible_df):,}")
            d3.metric("Rows excluded", f"{len(source_df) - len(eligible_df):,}")
            d4.metric("Canonical fields", str(len(FIELDS)))
            eligible_pct = 100 * len(eligible_df) / len(source_df) if len(source_df) else 0
            st.progress(eligible_pct / 100 if source_df.shape[0] else 0)
            st.caption(f"Eligibility retention: {eligible_pct:.1f}%. The rules are applied before the controlled corruptions are generated.")

            with st.expander("Review eligibility rules and pass counts", expanded=False):
                rules_df = pd.DataFrame({"Inclusion rule": list(eligibility_counts.keys()), "Rows passing rule": list(eligibility_counts.values())})
                st.dataframe(rules_df, use_container_width=True, hide_index=True)
                st.caption("These counts are independent per rule, so they do not add up to the final eligible count.")

            p1, p2 = st.columns([1.1, 1.4])
            with p1:
                st.markdown("#### Build benchmark")
                sample_size = st.selectbox("Records to sample", [20, 50, 100], index=2, key="sample_size")
                seed = st.number_input("Random seed", min_value=1, max_value=999999, value=int(st.session_state.seed), step=1, key="sample_seed")
                st.caption("Use the same seed and pool for reproducibility. Building a new benchmark clears prior pipeline outputs.")
                if st.button("Build / reset controlled benchmark", type="primary", use_container_width=True, disabled=len(eligible_df) < int(sample_size)):
                    sampled = eligible_df.sample(n=int(sample_size), random_state=int(seed)).reset_index(drop=True)
                    gold_records: dict[str, dict[str, Any]] = {}
                    corrupted_records: dict[str, dict[str, Any]] = {}
                    for i, row in sampled.iterrows():
                        record_key = f"R{i + 1:05d}"
                        gold = canonicalize(row)
                        gold_records[record_key] = {k: json_safe(v) for k, v in gold.items()}
                        corrupted = corrupt_record(gold, i)
                        corrupted["record_key"] = record_key
                        corrupted_records[record_key] = {k: json_safe(v) for k, v in corrupted.items()}
                    reset_benchmark()
                    st.session_state.gold_records = gold_records
                    st.session_state.corrupted_records = corrupted_records
                    st.session_state.seed = int(seed)
                    st.session_state.benchmark_size = int(sample_size)
                    st.session_state.benchmark_built_at = datetime.now(timezone.utc).isoformat()
                    st.session_state.benchmark_fingerprint = st.session_state.source_fingerprint
                    st.success(f"Benchmark created: {sample_size} records with pre-corruption ground truth preserved.")
                    st.rerun()
            with p2:
                st.markdown("#### Preview source records")
                st.dataframe(source_df.head(8), use_container_width=True, hide_index=True)
                st.caption(f"File: `{uploaded.name}` · SHA-256: `{hashlib.sha256(file_bytes or b'').hexdigest()[:16]}…`")
                st.markdown("#### Required fields")
                st.code(", ".join(FIELDS), language="text")

            if st.session_state.gold_records:
                st.success(f"Current benchmark is ready: {st.session_state.benchmark_size} records · seed {st.session_state.seed}.")
                with st.expander("Inspect one target record and its controlled corrupted form"):
                    keys = list(st.session_state.gold_records)
                    selected_key = st.selectbox("Benchmark record", keys, key="source_preview_record")
                    v1, v2 = st.columns(2)
                    with v1:
                        st.markdown("**Ground truth (canonical)**")
                        st.json(st.session_state.gold_records[selected_key])
                    with v2:
                        st.markdown("**Input to both pipelines**")
                        st.json(st.session_state.corrupted_records[selected_key])

    with tab_run:
        st.subheader("2 · Run both ETL pipelines")
        if not st.session_state.gold_records:
            st.info("First upload `train.csv` and build the benchmark under **Dataset & benchmark**.")
        else:
            keys = list(st.session_state.gold_records)
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.markdown("#### A · Rule-based baseline")
                st.write("Deterministic Python rules parse numeric strings, normalize flags, standardize categorical values and remove the controlled order-ID prefix.")
                if st.button("Run rule-based ETL", type="primary", use_container_width=True):
                    outputs: dict[str, dict[str, Any]] = {}
                    runtimes: dict[str, float] = {}
                    for key in keys:
                        t0 = time.perf_counter()
                        outputs[key] = rule_transform(st.session_state.corrupted_records[key])
                        runtimes[key] = time.perf_counter() - t0
                    st.session_state.rule_outputs = outputs
                    st.session_state.rule_times = runtimes
                    st.success(f"Rule-based run completed for {len(outputs)} records.")
                if st.session_state.rule_outputs:
                    st.metric("Records processed", f"{len(st.session_state.rule_outputs):,}")
                    baseline = evaluate_outputs(st.session_state.gold_records, st.session_state.rule_outputs, keys, "Rule-based")
                    st.metric("Field-level accuracy", f"{baseline['field_accuracy_pct']:.2f}%")
            with rcol2:
                st.markdown("#### B · Groq-hosted LLM pipeline")
                st.write("The selected model receives the same controlled inputs and must return all target fields as structured JSON.")
                if api_key:
                    st.success(f"API key detected · model `{model_id}`")
                else:
                    st.warning("Configure `GROQ_API_KEY` in Streamlit Secrets before calling the model.")
                target_options = [n for n in (20, 50, 100) if n <= len(keys)]
                default_target = min(20, len(keys))
                target_n = st.select_slider("Run LLM for the first N records", options=target_options, value=default_target, key="llm_target_n")
                batch_size = st.selectbox("Records per API request", [1, 5], index=1, key="llm_batch_size")
                st.caption("Run 20 first, then increase to 50 and 100. The app saves responses in the active Streamlit session; download the audit export after each important run.")
                if st.button("Run / extend LLM experiment", type="primary", use_container_width=True, disabled=not bool(api_key)):
                    try:
                        client = Groq(api_key=api_key)
                        target_keys = keys[:int(target_n)]
                        missing_keys = [key for key in target_keys if key not in st.session_state.llm_outputs]
                        if not missing_keys:
                            st.info("The selected records already have LLM output. Build a new benchmark to repeat the run, or choose a larger N to extend it.")
                        else:
                            batches = [missing_keys[i:i + int(batch_size)] for i in range(0, len(missing_keys), int(batch_size))]
                            progress = st.progress(0.0)
                            status = st.empty()
                            for batch_index, batch_keys in enumerate(batches, start=1):
                                batch_rows = []
                                for key in batch_keys:
                                    record = dict(st.session_state.corrupted_records[key])
                                    record["record_key"] = key
                                    batch_rows.append(record)
                                call_start = time.perf_counter()
                                try:
                                    output_map, meta = call_groq(
                                        client, batch_rows, model_id,
                                        float(st.session_state.input_rate), float(st.session_state.output_rate),
                                    )
                                except Exception as exc:
                                    meta = failed_call_meta(batch_rows, model_id, time.perf_counter() - call_start, exc)
                                    output_map = {}
                                st.session_state.llm_calls.append(meta)
                                for key in batch_keys:
                                    output = output_map.get(key, {})
                                    st.session_state.llm_outputs[key] = output
                                    st.session_state.llm_meta_by_key[key] = meta
                                status.write(f"Batch {batch_index}/{len(batches)} · requested {len(batch_keys)} records · returned {sum(bool(output_map.get(k)) for k in batch_keys)} record objects")
                                progress.progress(batch_index / len(batches))
                            errors = sum(1 for call in st.session_state.llm_calls if call.get("error"))
                            st.success(f"Run finished. LLM output is stored for {len(st.session_state.llm_outputs)} records; total logged API errors: {errors}.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Could not initialize the Groq client: {type(exc).__name__}: {str(exc)[:350]}")

            st.markdown("---")
            st.markdown("#### Run status")
            s1, s2, s3 = st.columns(3)
            s1.metric("Benchmark records", f"{len(keys):,}")
            s2.metric("Rule outputs", f"{len(st.session_state.rule_outputs):,}")
            s3.metric("LLM outputs", f"{len(st.session_state.llm_outputs):,}")
            st.caption("The model is not expected to be superior by design: the rule-based baseline is explicitly programmed for the known, reversible transformations. That is a useful baseline for deciding whether LLM complexity is justified in this bounded task.")

    with tab_results:
        st.subheader("3 · Compare measured outcomes")
        if not st.session_state.gold_records or not st.session_state.rule_outputs:
            st.info("Build a benchmark and run the rule-based baseline first. Run the LLM pipeline to show the comparison.")
        elif not st.session_state.llm_outputs:
            baseline = evaluate_outputs(st.session_state.gold_records, st.session_state.rule_outputs, list(st.session_state.gold_records), "Rule-based")
            c1, c2 = st.columns(2)
            c1.metric("Rule field accuracy", f"{baseline['field_accuracy_pct']:.2f}%")
            c2.metric("Rule record accuracy", f"{baseline['record_accuracy_pct']:.2f}%")
            st.info("No LLM outputs yet. Go to **Run pipelines** and run the model on the first 20 records to enable the paired comparison.")
        else:
            summary_df, field_df = build_result_summary()
            if summary_df.empty:
                st.info("Run an LLM sample of 20, 50 or 100 records to build comparable result rows.")
            else:
                completed_sizes = sorted(summary_df["sample_size"].unique())
                largest_n = int(max(completed_sizes))
                largest_rows = summary_df[summary_df["sample_size"] == largest_n]
                by_method = {row["method"]: row for _, row in largest_rows.iterrows()}
                a, b, c, d = st.columns(4)
                a.metric("Records compared", f"{largest_n:,}")
                a_delta = float(by_method["LLM-assisted"]["field_accuracy_pct"] - by_method["Rule-based"]["field_accuracy_pct"])
                b.metric("Rule field accuracy", f"{by_method['Rule-based']['field_accuracy_pct']:.2f}%")
                c.metric("LLM field accuracy", f"{by_method['LLM-assisted']['field_accuracy_pct']:.2f}%", delta=f"{a_delta:+.2f} pp")
                llm_cost = by_method["LLM-assisted"]["estimated_service_cost_usd"]
                d.metric("LLM service-equivalent cost", f"${llm_cost:.6f}" if pd.notna(llm_cost) else "Unavailable")
                st.caption("Cost is token-count × reference unit price, not a billing statement. A provider free allowance can make actual expenditure $0.")

                chart1, chart2 = st.columns(2)
                with chart1:
                    acc_long = summary_df.melt(
                        id_vars=["sample_size", "method"],
                        value_vars=["field_accuracy_pct", "record_accuracy_pct"],
                        var_name="measure", value_name="accuracy_pct",
                    )
                    acc_long["measure"] = acc_long["measure"].map({"field_accuracy_pct": "Field accuracy", "record_accuracy_pct": "Complete-record accuracy"})
                    fig = px.bar(acc_long, x="measure", y="accuracy_pct", color="method", barmode="group", facet_col="sample_size", title="Accuracy on completed sample sizes", range_y=[0, 100], labels={"measure": "Measure", "accuracy_pct": "Accuracy (%)", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 400), use_container_width=True)
                with chart2:
                    fig = px.line(summary_df, x="sample_size", y="processing_seconds", color="method", markers=True, title="Observed processing time", labels={"sample_size": "Records tested", "processing_seconds": "Observed seconds", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 400), use_container_width=True)

                chart3, chart4 = st.columns(2)
                with chart3:
                    cost_df = summary_df[summary_df["method"] == "LLM-assisted"].copy()
                    if not cost_df.empty and cost_df["estimated_service_cost_usd"].notna().any():
                        fig = px.bar(cost_df, x="sample_size", y="estimated_service_cost_usd", title="LLM service-equivalent cost by tested sample size", labels={"sample_size": "Records actually tested", "estimated_service_cost_usd": "Estimated cost (USD)"})
                        st.plotly_chart(style_plot(fig, 330), use_container_width=True)
                    else:
                        st.info("Token usage was not returned for these calls, so cost estimates are unavailable.")
                with chart4:
                    throughput_df = summary_df.copy()
                    fig = px.line(throughput_df, x="sample_size", y="records_per_second", color="method", markers=True, title="Observed throughput", labels={"sample_size": "Records actually tested", "records_per_second": "Records / second", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 330), use_container_width=True)

                st.markdown("#### Summary table")
                display_cols = ["sample_size", "method", "field_accuracy_pct", "record_accuracy_pct", "processing_seconds", "records_per_second", "input_tokens", "output_tokens", "estimated_service_cost_usd", "attempted_api_calls", "api_errors"]
                st.dataframe(summary_df[display_cols].round(6), use_container_width=True, hide_index=True)
                st.caption("Times are measured, not extrapolated. Rule time measures local Python transformation time; LLM time sums API request wall time, including provider/network response. This is not a fully matched hardware/runtime comparison.")

                if not field_df.empty:
                    st.markdown(f"#### Field-level accuracy · largest completed sample ({largest_n} records)")
                    fig = px.bar(field_df, x="accuracy_pct", y="field", color="method", barmode="group", orientation="h", range_x=[0, 100], labels={"accuracy_pct": "Correct outputs (%)", "field": "Canonical field", "method": "Pipeline"})
                    st.plotly_chart(style_plot(fig, 620), use_container_width=True)

                with st.expander("Inspect model errors and missing/incorrect responses"):
                    errors = [call for call in st.session_state.llm_calls if call.get("error")]
                    if errors:
                        st.dataframe(pd.DataFrame([{k: call.get(k) for k in ["call_id", "timestamp_utc", "model_id", "record_keys", "elapsed_sec", "error"]} for call in errors]), use_container_width=True, hide_index=True)
                    else:
                        st.success("No API exceptions were logged.")
                    st.caption("A response that is valid JSON but has wrong or missing fields is still scored as incorrect for those fields.")

    with tab_audit:
        st.subheader("4 · Export reproducible evidence")
        st.write("Download the audit JSON after each successful run. Keep this export outside the public GitHub repository; it contains sampled source-derived records and experimental outputs.")
        has_benchmark = bool(st.session_state.gold_records)
        audit = make_audit_payload()
        st.download_button(
            "⬇️ Download complete audit trail (JSON)",
            data=json.dumps(audit, ensure_ascii=False, indent=2, default=str),
            file_name="icitiit27_etl_experiment_audit.json",
            mime="application/json",
            disabled=not has_benchmark,
            use_container_width=True,
        )
        if has_benchmark:
            rows: list[dict[str, Any]] = []
            for key, gold in st.session_state.gold_records.items():
                rule_out = st.session_state.rule_outputs.get(key, {})
                llm_out = st.session_state.llm_outputs.get(key, {})
                item: dict[str, Any] = {
                    "record_key": key,
                    "gold_json": json.dumps(gold, ensure_ascii=False),
                    "corrupted_input_json": json.dumps(st.session_state.corrupted_records[key], ensure_ascii=False),
                    "rule_output_json": json.dumps(rule_out, ensure_ascii=False),
                    "llm_output_json": json.dumps(llm_out, ensure_ascii=False),
                    "rule_processing_seconds": st.session_state.rule_times.get(key),
                    "llm_call_id": st.session_state.llm_meta_by_key.get(key, {}).get("call_id"),
                }
                rows.append(item)
            st.download_button(
                "⬇️ Download record-level outputs (CSV)",
                data=pd.DataFrame(rows).to_csv(index=False),
                file_name="icitiit27_etl_record_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Build a benchmark before exporting record-level outputs.")

        st.markdown("#### Run metadata")
        metadata = audit["metadata"]
        meta_display = {k: metadata.get(k) for k in ["app_version", "research_title", "dataset_title", "dataset_source_url", "uploaded_filename", "uploaded_file_sha256", "uploaded_rows", "eligible_rows", "model_id", "random_seed", "benchmark_size", "benchmark_built_at_utc", "exported_at_utc"]}
        st.json(meta_display)
        st.warning("Never commit the API key, raw dataset, audit JSON or record-level CSV to a public GitHub repository. These files are experimental evidence and must be backed up privately.")

    with tab_protocol:
        st.subheader("5 · Research protocol and limitations")
        st.markdown("#### Experimental comparison")
        protocol_rows = [
            {"Component": "Dataset", "Protocol": f"Kaggle competition training file — [{DATASET_TITLE}]({DATA_SOURCE_URL})"},
            {"Component": "Analysis task", "Protocol": "Normalize controlled formatting and type variations; not return prediction"},
            {"Component": "Ground truth", "Protocol": "Canonicalized eligible input records captured before controlled corruption"},
            {"Component": "Rule-based system", "Protocol": "Explicit Python rules for numeric parsing, flags, case/whitespace and order-ID prefix"},
            {"Component": "LLM system", "Protocol": f"Groq API model `{model_id}`, temperature 0, JSON-mode response"},
            {"Component": "Primary accuracy metrics", "Protocol": "Field-level exact-within-tolerance accuracy and complete-record accuracy"},
            {"Component": "Efficiency metrics", "Protocol": "Measured processing wall time, records/second, token count and reference-priced service-equivalent cost"},
            {"Component": "Scale tests", "Protocol": "Only sample sizes actually run (20, 50, 100); no linear extrapolation"},
        ]
        st.dataframe(pd.DataFrame(protocol_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Limitations to state in the paper")
        st.markdown(
            "- The Kaggle listing describes transactions covering 2022–2024, but the original retailer and real-world provenance are not independently verified. Do not describe the data as confirmed small-business transactions.\n"
            "- This benchmark uses eligible source rows and adds controlled, reversible formatting/type variations. It does not test semantic correction of unknown errors, missing-value imputation, schema mapping between unrelated tables, or real production ingestion.\n"
            "- Because the transformation errors are known and reversible, a well-designed rule-based baseline is expected to perform strongly. The experiment is useful for testing when an LLM adds value, cost or latency in this bounded setting—not for assuming LLM superiority.\n"
            "- The reference cost excludes hosting, local compute, engineering time and any other operational expenses. Provider price/allowance must be verified on the experiment date.\n"
            "- A single sample run is exploratory. For a defensible paper, preserve audit files, record model/date/code version, repeat with predeclared seeds where practical, and report failures rather than silently discarding them."
        )
        st.markdown("#### Conference reference")
        st.markdown("ICITIIT 2027 author instructions: [official submission guidelines](https://icitiit27.iiitkottayam.ac.in/instructions-to-authors/). Confirm the current deadline, page limit and formatting requirements on the official site before submission.")


if __name__ == "__main__":
    main()
