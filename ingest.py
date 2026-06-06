import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from config import CONFIG
from utils import logger

STANDARD_EMPLOYEE_SCHEMA = CONFIG["employee_schema"]


# ── Dead-letter log 
_dead_letters: list = []


def _log_dead_letter(source: str, reason: str, raw=None) -> None:
    """Record a skipped item to the dead-letter log and emit a warning."""
    entry = {"source": source, "reason": reason, "raw": str(raw)[:300]}
    _dead_letters.append(entry)
    logger.warning(f"  [DEAD LETTER] {source}: {reason}")


def get_dead_letters() -> pd.DataFrame:
    """Return all dead-letter records accumulated during the current run."""
    return pd.DataFrame(_dead_letters) if _dead_letters else pd.DataFrame(
        columns=["source", "reason", "raw"]
    )


# Source 1: GlobalTech HRIS (CSV)

def ingest_globaltech_hris(filepath: Path) -> pd.DataFrame:
    """
    Load the GlobalTech HRIS CSV into a DataFrame.
    """
    if not filepath.exists():
        _log_dead_letter("globaltech_hris", f"File not found: {filepath}")
        return pd.DataFrame(columns=STANDARD_EMPLOYEE_SCHEMA)

    try:
        df = pd.read_csv(
            filepath,
            dtype={"employee_id": str, "manager_id": str},
            na_values=["", "N/A", "null", "NULL", "none", "NaN"],
        )
    except Exception as exc:
        _log_dead_letter("globaltech_hris", f"Failed to read CSV: {exc}")
        return pd.DataFrame(columns=STANDARD_EMPLOYEE_SCHEMA)

    df["source_system"]  = "globaltech_hris"
    df["company_origin"] = "GlobalTech"

    logger.info(f"  [globaltech_hris] {len(df):,} records  ←  {filepath.name}")
    return df

# Source 2: AcquiredCo JSON (paginated API simulation)

def ingest_acquiredco_json(
    filepath: Path,
    page_size: int = CONFIG["api_page_size"],
) -> pd.DataFrame:
    """Load AcquiredCo employee data from JSON, simulating a paginated REST API."""
    if not filepath.exists():
        _log_dead_letter("acquiredco_api", f"File not found: {filepath}")
        return pd.DataFrame(columns=STANDARD_EMPLOYEE_SCHEMA)

    try:
        payload = json.loads(filepath.read_text(encoding="utf-8"))
        all_employees = payload["employees"]
        total_records = payload.get("total_records", len(all_employees))
        total_pages = math.ceil(total_records / page_size)
    except Exception as exc:
        _log_dead_letter("acquiredco_api", f"Failed to parse JSON: {exc}")
        return pd.DataFrame(columns=STANDARD_EMPLOYEE_SCHEMA)

    logger.info("Ingesting AcquiredCo API (paginated)...")

    all_records = []
    page = 1

    while True:
        start = (page - 1) * page_size
        batch = all_employees[start : start + page_size]

        if not batch:
            break

        all_records.extend(batch)
        logger.info(f"  Fetched page {page} ({len(batch)} records)")
        page += 1

        if page > total_pages:
            break

    df = pd.json_normalize(all_records, sep="_")

    df = df.rename(columns={
        "employee_identifier":       "employee_id",
        "name_first":                "first_name",
        "name_last":                 "last_name",
        "contact_email":             "email",
        "assignment_department":     "department",
        "assignment_role":           "job_title",
        "assignment_hire_timestamp": "hire_date",
        "assignment_location":       "country",
        "employment_type":           "employment_type",
        "manager_employee_id":       "manager_id",
    })

    df["source_system"]  = "acquiredco_api"
    df["company_origin"] = "AcquiredCo"

    logger.info(f"  [acquiredco_api] {len(df):,} records  ←  {filepath.name}")
    return df


# Source 3: Benefits Enrollment (XML)

def ingest_benefits_xml(filepath: Path) -> pd.DataFrame:
    """
    Parse the benefits enrollment XML using Python's built-in ElementTree.
    """
    if not filepath.exists():
        _log_dead_letter("benefits_xml", f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        records = []
        for enrollment in root.findall("enrollment"):
            records.append({
                "employee_id":      enrollment.findtext("employee_id"),
                "plan_type":        enrollment.findtext("plan_type"),
                "coverage_level":   enrollment.findtext("coverage_level"),
                "enrollment_date":  enrollment.findtext("enrollment_date"),
                "premium_employee": enrollment.findtext("premium_employee"),
                "premium_employer": enrollment.findtext("premium_employer"),
            })

    except ET.ParseError as exc:
        _log_dead_letter("benefits_xml", f"XML parse error: {exc}")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    logger.info(f"  [benefits_xml] {len(df):,} enrollment records  ←  {filepath.name}")
    return df

# Source 4: Payroll (Excel)

def ingest_payroll_xlsx(filepath: Path) -> pd.DataFrame:
    """
    Load the combined payroll Excel file into a DataFrame.
    """
    if not filepath.exists():
        _log_dead_letter("payroll_xlsx", f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(
            filepath,
            dtype={"employee_id": str, "base_salary": str},
        )
    except Exception as exc:
        _log_dead_letter("payroll_xlsx", f"Failed to read Excel: {exc}")
        return pd.DataFrame()

    logger.info(f"  [payroll_xlsx] {len(df):,} payroll records  ←  {filepath.name}")
    return df


# Schema alignment

def align_to_employee_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a source DataFrame contains all standard employee schema columns.
    """
    for col in STANDARD_EMPLOYEE_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    extras = [c for c in df.columns if c not in STANDARD_EMPLOYEE_SCHEMA]
    return df[STANDARD_EMPLOYEE_SCHEMA + extras].copy()


# Entry point

def ingest_all_sources() -> dict:
    """
    Ingest all 4 data sources and return a dict of DataFrames.
    """
    logger.info("Ingestion: GlobalTech HRIS (CSV)")
    hris_df = ingest_globaltech_hris(CONFIG["input_dir"] / "globaltech_hris.csv")

    logger.info("Ingestion: AcquiredCo API (JSON, paginated)")
    acquiredco_df = ingest_acquiredco_json(CONFIG["input_dir"] / "acquiredco_api.json")


    logger.info("Ingestion: Benefits Enrollment (XML)")
    benefits_df = ingest_benefits_xml(CONFIG["input_dir"] / "benefits_enrollment.xml")

    logger.info("Ingestion: Payroll (Excel)")
    payroll_df = ingest_payroll_xlsx(CONFIG["input_dir"] / "payroll_data.xlsx")

    # Align employee sources to the standard schema, then concatenate
    hris_aligned       = align_to_employee_schema(hris_df)
    acquiredco_aligned = align_to_employee_schema(acquiredco_df)
    employees_df       = pd.concat([hris_aligned, acquiredco_aligned], ignore_index=True)

    # Per-source record count summary
    logger.info("Ingestion summary")
    for origin, count in employees_df["company_origin"].value_counts().items():
        logger.info(f"  {origin:20s}: {count:,} employee records")
    logger.info(f"  {'Total employees':20s}: {len(employees_df):,}")
    logger.info(f"  {'Payroll records':20s}: {len(payroll_df):,}")
    logger.info(f"  {'Benefits records':20s}: {len(benefits_df):,}")

    dead = get_dead_letters()
    if len(dead) > 0:
        logger.warning(f"  Dead-letter total    : {len(dead)} skipped records")

    return {
        "employees": employees_df,
        "payroll":   payroll_df,
        "benefits":  benefits_df,
    }
