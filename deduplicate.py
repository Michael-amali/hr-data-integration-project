"""
Deduplication module for the GlobalTech HR Data Integration pipeline.

Three-pass deduplication strategy:
  Pass 1 — Exact employee ID match
    Records sharing the same namespaced ID (GT-XXXXXX or AC-XXXXXX) are the
    same employee. Keep the highest-priority source; remove the rest.
    Priority: globaltech_hris (1) > payroll (2) > benefits (3) > acquiredco_api (4)

  Pass 2 — Email match (all records)
    Records sharing the same email are the same person (e.g., contractor in
    both companies). Keep the highest-priority source; remove the rest.

  Pass 3 — Fuzzy name + hire date match
    Use rapidfuzz to compare full name pairs with similarity >= 88%.
    Blocking: records are sorted by hire_date; only records within 30 days of
    each other are compared. This keeps the algorithm O(n × k), not O(n²).
    These are flagged as probable_match — HR must confirm via the review file.
    No records are auto-removed in this pass.

Additional outputs:
  Ghost employees: payroll records with no matching employee_id in the employee DF.
  Provenance:      source_systems and dedup_method columns on every golden record.

Entry point: run_deduplication(data: dict) -> dict
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from config import CONFIG
from utils import logger


# ── Pass 1: Exact employee ID match ──────────────────────────────────────────

def pass1_exact_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records that share the same namespaced employee_id.

    When multiple records share an ID, the highest-priority source survives.
    Source priority (from config): globaltech_hris=1, acquiredco_api=4.

    Catches ACQ_DUP_XXXXX records: they were namespaced to the same AC-XXXXXX
    as their ACQ_XXXXX counterpart in clean.py, making them exact-ID duplicates.

    Adds columns:
        _priority      — numeric source priority (dropped before final export)
        source_systems — comma-joined sources for all records sharing this ID
        dedup_method   — "exact_id" if a duplicate existed; "single_source" if not
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


# ── Pass 2: Email match ───────────────────────────────────────────────────────

def pass2_email_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records sharing the same email address.

    When multiple records share an email, the highest-priority source survives
    (already sorted by _priority from Pass 1). The surviving record's
    source_systems is updated to include all sources from the duplicate group.

    Applies across all records (GT-GT, GT-AC, AC-AC).
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
    email_source_map = (
        df[df["email"].isin(dup_emails)]
        .groupby("email")
        .apply(lambda g: ",".join(sorted(set(",".join(g["source_systems"]).split(",")))))
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


# ── Pass 3: Fuzzy name + hire date ───────────────────────────────────────────

def pass3_fuzzy_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find probable duplicates using fuzzy name matching, blocked by hire date.

    Algorithm:
      1. Drop records with null hire_date or name fields.
      2. Sort remaining records by hire_date (ascending).
      3. For each record i, advance j forward while hire_date[j] - hire_date[i]
         <= 30 days. Compare names within this window.
      4. Pairs with rapidfuzz token_sort_ratio >= 88 are flagged.

    The hire_date block keeps complexity at O(n × k) where k is the average
    number of records within the 30-day window — far less than O(n²).

    Returns a DataFrame of probable match pairs for HR review.
    Does NOT remove any records from the main employee dataset.
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


# ── Ghost employee detection ──────────────────────────────────────────────────

def detect_ghosts(payroll: pd.DataFrame, original_employee_ids: set) -> pd.DataFrame:
    """
    Find payroll records with no matching employee_id in the original HRIS.

    Compares against the FULL pre-dedup employee ID set, not the post-dedup golden
    set. This is critical: an employee removed by email dedup is not a ghost —
    they exist in the HRIS. A ghost is someone in payroll with NO HRIS record at all.

    Ghost flag reasons:
      "ghost_prefix_id"  — employee_id starts with GHOST_ (pre-seeded test case)
      "no_hris_match"    — employee_id not found in any HRIS record

    Required output columns: payroll_employee_id, name, salary_usd_annual, ghost_flag_reason
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


# ── Payroll merge ─────────────────────────────────────────────────────────────

def merge_payroll(golden: pd.DataFrame, payroll: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the most recent payroll record onto each employee.

    For employees with multiple payroll rows (multiple pay periods), only the
    row with the latest effective_date is kept. Employees with no payroll
    record retain NaN in all salary columns.

    Updates source_systems to append ",payroll" where payroll data was found.
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


# ── Entry point ───────────────────────────────────────────────────────────────

def run_deduplication(data: dict) -> dict:
    """
    Run all deduplication passes and produce the golden dataset and side outputs.

    Parameters
    ----------
    data : dict
        Output of clean_all():
        {"employees": DataFrame, "payroll": DataFrame, "benefits": DataFrame}

    Returns
    -------
    dict with keys:
        "golden"           — cleaned, deduplicated employees with payroll merged in
        "ghosts"           — payroll records with no HRIS match (ghost employee report)
        "probable_matches" — fuzzy-matched pairs for HR review
        "benefits"         — passed through unchanged for visualization
    """
    employees = data["employees"].copy()
    payroll   = data["payroll"].copy()

    # Snapshot ALL employee IDs before any dedup pass.
    # Ghost detection uses this set so that employees merged away by email dedup
    # are NOT flagged as ghosts — they exist in HRIS, just not in the golden set.
    original_employee_ids = set(employees["employee_id"].dropna().astype(str))

    logger.info(f"  Starting dedup: {len(employees):,} employee records")

    # ── Pass 1: Exact ID ─────────────────────────────────────────────────────
    employees = pass1_exact_id(employees)

    # ── Pass 2: Email match ──────────────────────────────────────────────────
    employees = pass2_email_match(employees)

    # ── Pass 3: Fuzzy name + hire date (review file only, no records removed) ──
    probable_matches = pass3_fuzzy_match(employees)

    # ── Ghost detection ───────────────────────────────────────────────────────
    # Use the full PRE-DEDUP employee ID set so that employees removed by email
    # dedup are not incorrectly flagged as ghosts. A ghost is someone in payroll
    # who has NO HRIS record at all — not someone whose record was merged away.
    ghosts = detect_ghosts(payroll, original_employee_ids)

    # ── Merge payroll ─────────────────────────────────────────────────────────
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
