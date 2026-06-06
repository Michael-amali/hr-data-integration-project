import unicodedata

import numpy as np
import pandas as pd

from config import CONFIG
from utils import logger


# Name standardization

def standardize_name(series: pd.Series) -> pd.Series:
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

# Department standardization

def standardize_department(series:pd.Series) -> pd.Series:
        return (
        series
        .astype(str)
        .map(CONFIG['employment_type_map'])
    )


# Employee ID namespacing

def _namespace_id(raw_id: str, company_origin: str) -> str:
    """
    Convert a raw employee ID to namespaced GT-XXXXXX / AC-XXXXXX format.
    """
    raw_id = str(raw_id).strip()

    if raw_id.upper().startswith("GHOST"):
        print(raw_id, company_origin)
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
        return f"GT-{int(raw_id):06d}"
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
        manager_id = row["manager_id"]
        if pd.isna(manager_id) or str(manager_id).strip() in ("", "nan", "None"):
            return np.nan
        return _namespace_id(str(manager_id), row["company_origin"])

    df["manager_id"] = df.apply(_ns_manager, axis=1)
    return df


# Hire date normalization

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

# Email standardization
def standardize_emails(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )

# Salary cleaning

def extract_numeric_salary(series: pd.Series) -> pd.Series:
    """
    Strip currency symbols ($, £, €) and thousand-separator commas, return float.
    """
    cleaned = series.astype(str).str.replace(r"[$£€,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def compute_salary_usd_annual(
    base_salary: pd.Series,
    currency: pd.Series,
    pay_frequency: pd.Series,
) -> pd.Series:
    """
    Convert base salary to annualised USD.
    """
    freq_multipliers = pay_frequency.map(CONFIG["pay_frequency_multipliers"]).fillna(1)
    fx_rates         = currency.map(CONFIG["fx_rates"]).fillna(1.0)
    return (base_salary * freq_multipliers * fx_rates).round(2)


def clean_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the payroll DataFrame.
    """
    df = df.copy()

    df["employee_id"] = [
        _namespace_id(eid, src)
        for eid, src in zip(df["employee_id"].astype(str), df["source"].astype(str))
    ]

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


# Employee DataFrame cleaning

def clean_employees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the employees DataFrame.
    """
    df = df.copy()

    df["first_name"] = standardize_name(df["first_name"])
    df["last_name"]  = standardize_name(df["last_name"])

    df = namespace_employee_ids(df)

    df["employment_type"] = standardize_department(df["employment_type"])
    df["email"] = standardize_emails(df["email"])
    df = standardize_hire_date(df)


    logger.info(f"  Names standardized: {len(df):,} records")
    gt_count = df["employee_id"].astype(str).str.startswith("GT-").sum()
    ac_count = df["employee_id"].astype(str).str.startswith("AC-").sum()
    logger.info(f"  Employee IDs namespaced: {gt_count:,} GT- | {ac_count:,} AC-")

    logger.info(f"  Records after cleaning: {len(df):,}")
    return df


# Entry point

def clean_all(data: dict) -> dict:
    """
    Clean all DataFrames in the ingest output dict.
    """
    logger.info("-- Cleaning: Employees ----------------------------------------")
    employees_clean = clean_employees(data["employees"])

    logger.info("-- Cleaning: Payroll ------------------------------------------")
    payroll_clean = clean_payroll(data["payroll"])

    return {
        "employees": employees_clean,
        "payroll":   payroll_clean,
        "benefits":  data["benefits"],
    }
