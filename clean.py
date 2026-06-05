"""
Data cleaning and transformation module.

Operations applied to the employees DataFrame:
  - Name standardization: NFC unicode normalization, title case,
    handles hyphens (Smith-Jones) and apostrophes (O'Brien) correctly
  - Employee ID namespacing: integers → GT-XXXXXX; ACQ_XXXXX → AC-XXXXXX
  - Manager ID namespacing: same rules as employee_id
  - Hire date normalization: → datetime64[ns]; out-of-range flagged
  - Email standardization: lowercase, strip whitespace

Operations applied to the payroll DataFrame:
  - Employee ID namespacing: same rules as employees
  - Base salary: strip currency symbols ($, £, €) and commas → numeric float
  - salary_usd_annual: base_salary × pay_frequency_multiplier × fx_rate
  - Original columns preserved as base_salary_original, currency_original,
    pay_frequency_original

Entry point: clean_all(data: dict) → dict
"""

import unicodedata

import numpy as np
import pandas as pd

from config import CONFIG
from utils import logger


# ── Name standardization ──────────────────────────────────────────────────────

def standardize_name(series: pd.Series) -> pd.Series:
    """
    Standardize a name column:
      1. NFC unicode normalization — ensures accented chars (Martínez, Müller,
         François) are in composed form for consistent storage and comparison
      2. Strip excess whitespace
      3. Title case — correctly capitalises O'Brien, Van Der Berg, Smith-Jones

    Note: Mc/Mac prefixes (McDonald) are not auto-corrected (out of scope).
    """
    def _clean(val: str) -> str:
        if not val or val.lower() in ("nan", "none"):
            return val
        nfc = unicodedata.normalize("NFC", val)
        return " ".join(w.capitalize() for w in nfc.strip().split())

    return (
        series
        .astype(str)
        .map(_clean)
        .str.strip()
        .replace({"Nan": np.nan, "None": np.nan, "": np.nan})
    )


# ── Employee ID namespacing ────────────────────────────────────────────────────

def _namespace_id(raw_id: str, company_origin: str) -> str:
    """
    Convert a raw employee ID to namespaced GT-XXXXXX / AC-XXXXXX format.

    Rules:
      GlobalTech integers  "14571"     → "GT-014571"
      AcquiredCo strings   "ACQ_00001" → "AC-000001"  (strips ACQ_ prefix)
      GHOST IDs            "GHOST_001" → unchanged     (handled at dedup stage)
      Unrecognised         unchanged, logged as warning
    """
    raw_id = str(raw_id).strip()

    if raw_id.upper().startswith("GHOST"):
        return raw_id

    # ACQ_DUP_XXXXX are intentional duplicate seeds; they map to the same AC-
    # namespace ID as their ACQ_XXXXX counterpart so Pass 1 dedup catches them.
    if raw_id.upper().startswith("ACQ_DUP_"):
        numeric = raw_id[8:].lstrip("0") or "0"
        return f"AC-{int(numeric):06d}"

    if raw_id.upper().startswith("ACQ_"):
        numeric = raw_id[4:].lstrip("0") or "0"
        return f"AC-{int(numeric):06d}"

    try:
        return f"GT-{int(float(raw_id)):06d}"
    except ValueError:
        logger.warning(f"  Could not namespace ID {raw_id!r} (origin={company_origin})")
        return raw_id


def namespace_employee_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply namespaced IDs to the employees DataFrame.

    Converts both employee_id and manager_id using company_origin as context.
    manager_id NaN values are preserved.
    """
    df["employee_id"] = [
        _namespace_id(eid, origin)
        for eid, origin in zip(
            df["employee_id"].astype(str), df["company_origin"].astype(str)
        )
    ]

    def _ns_manager(row):
        mid = row["manager_id"]
        if pd.isna(mid) or str(mid).strip() in ("", "nan", "None"):
            return np.nan
        return _namespace_id(str(mid), row["company_origin"])

    df["manager_id"] = df.apply(_ns_manager, axis=1)
    return df


# ── Hire date normalization ────────────────────────────────────────────────────

def standardize_hire_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize hire_date to datetime64[ns] and flag implausible values.

    Out-of-range definition: before 1970-01-01 or after today.
    Result stored in a new boolean column: hire_date_out_of_range.
    """
    today    = pd.Timestamp.today().normalize()
    min_date = pd.Timestamp(CONFIG["hire_date_min"])

    # The combined DF has three date formats:
    #   HRIS        "2016-09-21"                  (date only)
    #   AcquiredCo  "2024-06-27T00:00:00"         (ISO datetime, no tz)
    #   DUP records "2015-11-24T00:00:00Z"        (ISO datetime, UTC marker)
    # Strip trailing Z first, then use format='mixed' to handle all three.
    df["hire_date"] = pd.to_datetime(
        df["hire_date"].astype(str).str.rstrip("Z"),
        format="mixed",
        errors="coerce",
    )
    in_range = df["hire_date"].between(min_date, today)
    df["hire_date_out_of_range"] = ~in_range

    flagged = int(df["hire_date_out_of_range"].sum())
    if flagged:
        logger.info(f"  hire_date out-of-range flags: {flagged}")

    return df


# ── Salary cleaning ───────────────────────────────────────────────────────────

def extract_numeric_salary(series: pd.Series) -> pd.Series:
    """
    Strip currency symbols ($, £, €) and thousand-separator commas, return float.

    Examples: "$85,000" → 85000.0 | "£47,742" → 47742.0 | 77935 → 77935.0
    """
    cleaned = series.astype(str).str.replace(r"[$£€,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

    # return (
    #     series
    #     .astype(str)
    #     .str.replace(r"[$£€,\s]", "", regex=True)
    #     .pipe(pd.to_numeric, errors="coerce")
    # )


def compute_salary_usd_annual(
    base_salary: pd.Series,
    currency: pd.Series,
    pay_frequency: pd.Series,
) -> pd.Series:
    """
    Convert base salary to annualised USD.

    Formula: base_salary × pay_frequency_multiplier × fx_rate_to_usd

    Pay frequency multipliers (from config):
      Annual → ×1 | Monthly → ×12 | Bi-Weekly → ×26

    FX rates (fixed snapshot, from config):
      USD → 1.00 | EUR → 1.08 | GBP → 1.27
    """
    freq_multipliers = pay_frequency.map(CONFIG["pay_frequency_multipliers"]).fillna(1)
    fx_rates         = currency.map(CONFIG["fx_rates"]).fillna(1.0)
    return (base_salary * freq_multipliers * fx_rates).round(2)


def clean_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the payroll DataFrame.

    Steps:
      1. Namespace employee_id  (integers → GT-XXXXXX, ACQ_XXXXX → AC-XXXXXX)
      2. Preserve original salary/currency/frequency columns
      3. Extract numeric base_salary (strip $, £, €, commas)
      4. Compute salary_usd_annual

    New columns added:
      base_salary_original, currency_original, pay_frequency_original
      base_salary (numeric float), salary_usd_annual
    """
    df = df.copy()

    # Payroll IDs: numeric (GlobalTech) | ACQ_XXXXX (AcquiredCo) | GHOST_XXXX (unknown)
    # The "source" column in payroll indicates company: "GlobalTech" / "AcquiredCo"
    df["employee_id"] = [
        _namespace_id(eid, src)
        for eid, src in zip(df["employee_id"].astype(str), df["source"].astype(str))
    ]

    # Preserve originals before mutation
    df["base_salary_original"]   = df["base_salary"].copy()
    df["currency_original"]      = df["currency"].copy()
    df["pay_frequency_original"] = df["pay_frequency"].copy()

    # Strip symbols and commas → numeric float
    df["base_salary"] = extract_numeric_salary(df["base_salary"])

    # Annualise and convert to USD
    df["salary_usd_annual"] = compute_salary_usd_annual(
        df["base_salary"], df["currency"], df["pay_frequency"]
    )

    symbol_count = (
        df["base_salary_original"]
        .astype(str)
        .str.contains(r"[$£€]", regex=True, na=False)
        .sum()
    )
    logger.info(f"  Payroll: {symbol_count} salary strings with currency symbols cleaned")
    logger.info(
        f"  Payroll: salary_usd_annual computed for "
        f"{df['salary_usd_annual'].notna().sum():,} records"
    )
    return df


# ── Employee DataFrame cleaning ───────────────────────────────────────────────

def clean_employees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the employees DataFrame.

    Steps (in order):
      1. Name standardization   — NFC normalization + title case
      2. Employee / manager ID namespacing — GT-XXXXXX / AC-XXXXXX
      3. Hire date normalization — datetime64[ns] + hire_date_out_of_range flag
      4. Email standardization  — lowercase, strip whitespace

    Returns
    -------
    pd.DataFrame
        Cleaned employees DataFrame with hire_date_out_of_range added.
    """
    df = df.copy()

    df["first_name"] = standardize_name(df["first_name"])
    df["last_name"]  = standardize_name(df["last_name"])
    logger.info(f"  Names standardized: {len(df):,} records")

    df = namespace_employee_ids(df)
    gt_count = df["employee_id"].astype(str).str.startswith("GT-").sum()
    ac_count = df["employee_id"].astype(str).str.startswith("AC-").sum()
    logger.info(f"  Employee IDs namespaced: {gt_count:,} GT- | {ac_count:,} AC-")

    df = standardize_hire_date(df)

    df["email"] = (
        df["email"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )

    logger.info(f"  Records after cleaning: {len(df):,}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def clean_all(data: dict) -> dict:
    """
    Clean all DataFrames in the ingest output dict.

    Parameters
    ----------
    data : dict
        Output of ingest_all_sources():
        {"employees": DataFrame, "payroll": DataFrame, "benefits": DataFrame}

    Returns
    -------
    dict
        Same structure; employees and payroll are cleaned copies.
        benefits is passed through unchanged (no cleaning required).
    """
    logger.info("── Cleaning: Employees ────────────────────────────────────────")
    employees_clean = clean_employees(data["employees"])

    logger.info("── Cleaning: Payroll ──────────────────────────────────────────")
    payroll_clean = clean_payroll(data["payroll"])

    return {
        "employees": employees_clean,
        "payroll":   payroll_clean,
        "benefits":  data["benefits"],
    }
