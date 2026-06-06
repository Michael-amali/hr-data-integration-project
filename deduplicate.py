import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from config import CONFIG
from utils import logger


# Pass 1: Exact employee ID match

def pass1_exact_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records that share the same namespaced employee_id.
    """
    initial = len(df)

    df["_priority"] = df["source_system"].map(CONFIG["source_priority"]).fillna(99)
    df = df.sort_values("_priority").reset_index(drop=True)

    # Collect all sources for each employee_id group before dedup
    source_map = (
        df.groupby("employee_id")["source_system"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .to_dict()
    )

    ids_with_duplicates = set(df.loc[df.duplicated("employee_id", keep=False), "employee_id"])

    df = df.drop_duplicates(subset=["employee_id"], keep="first").copy()
    df["source_systems"] = df["employee_id"].map(source_map)
    df["dedup_method"]   = np.where(
        df["employee_id"].isin(ids_with_duplicates), "exact_id", "single_source"
    )

    removed = initial - len(df)
    logger.info(f"  Pass 1 (exact ID)  : {removed:,} duplicates removed → {len(df):,} records remain")
    return df


# Pass 2: Email match

def pass2_email_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records sharing the same email address.
    """
    initial = len(df)

    valid_email = df["email"].notna()
    dup_emails  = set(
        df.loc[valid_email & df.duplicated("email", keep=False), "email"]
    )

    if not dup_emails:
        logger.info(f"  Pass 2 (email)     : no duplicate emails found")
        return df

    # Collect merged source_systems for each duplicate email group
    def merge_sources(series):
        sources = ",".join(series).split(",")
        return ",".join(sorted(set(sources)))

    email_source_map = (
        df[df["email"].isin(dup_emails)]
        .groupby("email")["source_systems"]
        .apply(merge_sources)
        .to_dict()
    )

    non_dup_df = df[~df["email"].isin(dup_emails)].copy()

    dup_survivors = (
        df[df["email"].isin(dup_emails)]
        .drop_duplicates(subset=["email"], keep="first")
        .copy()
    )
    dup_survivors["source_systems"] = dup_survivors["email"].map(email_source_map)
    dup_survivors["dedup_method"]   = "email_match"

    df = pd.concat([non_dup_df, dup_survivors], ignore_index=True)

    removed = initial - len(df)
    logger.info(f"  Pass 2 (email)     : {removed:,} duplicates removed → {len(df):,} records remain")
    return df


# Pass 3: Fuzzy name + hire date

def pass3_fuzzy_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find probable duplicates using fuzzy name matching, blocked by hire date.
    """
    threshold = CONFIG["fuzzy_threshold"]       # 88
    window    = CONFIG["hire_date_window_days"] # 30

    # Work only on records with all required fields
    df_valid = (
        df.dropna(subset=["hire_date", "first_name", "last_name"])
        .copy()
    )
    df_valid["_full_name"] = (
        df_valid["first_name"].fillna("") + " " + df_valid["last_name"].fillna("")
    ).str.strip()
    df_valid = df_valid.sort_values("hire_date").reset_index(drop=True)

    hire_dates = df_valid["hire_date"].tolist()
    emp_ids    = df_valid["employee_id"].tolist()
    names      = df_valid["_full_name"].tolist()
    n          = len(df_valid)

    matches = []

    for i in range(n):
        j = i + 1
        while j < n:
            # hire_dates is sorted ascending, so once diff > window, all further j do too
            date_diff = (hire_dates[j] - hire_dates[i]).days
            if date_diff > window:
                break

            if emp_ids[i] == emp_ids[j]:
                j += 1
                continue

            score = fuzz.token_sort_ratio(names[i], names[j])
            if score >= threshold:
                matches.append({
                    "record_1_id":         emp_ids[i],
                    "record_2_id":         emp_ids[j],
                    "similarity_score":    round(float(score), 2),
                    "hire_date_diff_days": int(date_diff),
                    "recommended_action":  "manual_review",
                })
            j += 1

    result = pd.DataFrame(matches) if matches else pd.DataFrame(
        columns=["record_1_id", "record_2_id", "similarity_score",
                 "hire_date_diff_days", "recommended_action"]
    )

    logger.info(f"  Pass 3 (fuzzy)     : {len(result):,} probable match pairs flagged for HR review")
    return result


# Ghost employee detection

def detect_ghosts(payroll: pd.DataFrame, original_employee_ids: set) -> pd.DataFrame:
    """
    Find payroll records with no matching employee_id in the original HRIS.
    """
    ghost_mask = ~payroll["employee_id"].astype(str).isin(original_employee_ids)
    ghosts     = payroll[ghost_mask].copy()

    def _flag_reason(eid: str) -> str:
        return "ghost_prefix_id" if eid.upper().startswith("GHOST") else "no_hris_match"

    ghost_report = pd.DataFrame({
        "payroll_employee_id": ghosts["employee_id"].values,
        "name":                "Unknown",
        "salary_usd_annual":   ghosts["salary_usd_annual"].values,
        "ghost_flag_reason":   [_flag_reason(str(e)) for e in ghosts["employee_id"]],
    })

    by_reason = ghost_report["ghost_flag_reason"].value_counts().to_dict()
    logger.info(
        f"  Ghost employees    : {len(ghost_report):,} total  "
        + "  ".join(f"{r}: {c}" for r, c in by_reason.items())
    )
    return ghost_report


# Payroll merge
def merge_payroll(golden: pd.DataFrame, payroll: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the most recent payroll record onto each employee.
    """
    payroll = payroll.copy()
    payroll["effective_date"] = pd.to_datetime(payroll["effective_date"], errors="coerce")

    # Keep most recent payroll row per employee
    payroll_latest = (
        payroll
        .sort_values("effective_date", ascending=False)
        .drop_duplicates(subset=["employee_id"], keep="first")
    )

    salary_cols = [
        "employee_id", "base_salary", "currency", "pay_frequency",
        "bonus_target_pct", "effective_date", "salary_usd_annual",
        "base_salary_original", "currency_original", "pay_frequency_original",
    ]
    payroll_for_merge = payroll_latest[[c for c in salary_cols if c in payroll_latest.columns]]

    merged = golden.merge(payroll_for_merge, on="employee_id", how="left")

    has_payroll = merged["salary_usd_annual"].notna()
    merged.loc[has_payroll, "source_systems"] = (
        merged.loc[has_payroll, "source_systems"] + ",payroll"
    )

    logger.info(
        f"  Payroll merged     : {has_payroll.sum():,} / {len(merged):,} employees have payroll data"
    )
    return merged


# Entry point

def run_deduplication(data: dict) -> dict:
    """
    Run all deduplication passes and produce the golden dataset and side outputs.
    """
    employees = data["employees"].copy()
    payroll   = data["payroll"].copy()

    # ALL employee IDs before any dedup pass.
    original_employee_ids = set(employees["employee_id"].dropna().astype(str))

    logger.info(f"  Starting dedup: {len(employees):,} employee records")

    # Pass 1: Exact ID
    employees = pass1_exact_id(employees)

    # Pass 2: Email match
    employees = pass2_email_match(employees)

    # Pass 3: Fuzzy name + hire date (review file only, no records removed)
    probable_matches = pass3_fuzzy_match(employees)

    # Ghost detection
    ghosts = detect_ghosts(payroll, original_employee_ids)

    # Merge payroll
    golden = merge_payroll(employees, payroll)

    # Drop internal working column
    golden = golden.drop(columns=["_priority"], errors="ignore")

    logger.info(
        f"  Dedup complete: {len(employees):,} golden records  |  "
        f"{len(ghosts):,} ghosts  |  {len(probable_matches):,} probable matches"
    )

    return {
        "golden":           golden,
        "ghosts":           ghosts,
        "probable_matches": probable_matches,
        "benefits":         data["benefits"],
    }
