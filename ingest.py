"""
Multi-source ingestion module for the GlobalTech HR Data Integration pipeline.

Sources ingested:
  1. globaltech_hris.csv      — GlobalTech HRIS records (CSV)
  2. acquiredco_api.json      — AcquiredCo employee data (JSON, paginated API simulation)
  3. benefits_enrollment.xml  — Benefits enrollment records (XML)
  4. payroll_data.xlsx        — Combined payroll from both companies (Excel)

ingest_all_sources() returns a dict:
  {
    "employees": DataFrame,  # HRIS + AcquiredCo aligned to standard employee schema
    "payroll":   DataFrame,  # Payroll records — merged with employees at dedup stage
    "benefits":  DataFrame,  # Benefits enrollment records — joined for analytics/charts
  }

Dead-letter policy:
  Missing files or malformed records are logged and skipped. The pipeline never
  crashes from a bad input. Call get_dead_letters() to retrieve all skipped records.
"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pathlib import Path

from config import CONFIG
from utils import logger

STANDARD_EMPLOYEE_SCHEMA = CONFIG["employee_schema"]


# ── Dead-letter log ────────────────────────────────────────────────────────────

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


# ── Source 1: GlobalTech HRIS (CSV) ───────────────────────────────────────────

def ingest_globaltech_hris(filepath: Path) -> pd.DataFrame:
    """
    Load the GlobalTech HRIS CSV into a DataFrame.

    Parameters
    ----------
    filepath : Path
        Path to globaltech_hris.csv.

    Returns
    -------
    pd.DataFrame
        Employee records tagged with source_system and company_origin.

    Column mapping (CSV column → standard schema):
        employee_id     → employee_id      raw integer; namespaced to GT-XXXXXX in clean.py
        first_name      → first_name
        last_name       → last_name
        email           → email
        department      → department       as-sourced; no taxonomy normalization
        job_title       → job_title
        hire_date       → hire_date        normalized to datetime64 in clean.py
        country         → country
        employment_type → employment_type  already canonical: Full-Time / Part-Time / Contractor
        manager_id      → manager_id       raw integer; namespaced to GT-XXXXXX in clean.py
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


# ── Source 2: AcquiredCo JSON (paginated API simulation) ──────────────────────

def _parse_acquiredco_record(raw: dict) -> dict | None:
    """
    Parse one AcquiredCo employee record and return a flat dict.

    Returns None if the record is malformed (missing required keys).

    AcquiredCo JSON structure → standard schema mapping:
        employee_identifier          → employee_id      e.g. "ACQ_00001"; namespaced to AC-000001 in clean.py
        name.first                   → first_name
        name.last                    → last_name
        contact.email                → email
        assignment.department        → department       as-sourced; no taxonomy normalization
        assignment.role              → job_title
        assignment.hire_timestamp    → hire_date        ISO datetime string; normalized in clean.py
        assignment.location          → country
        employment.type              → employment_type  FT/PT/CONTRACTOR mapped to canonical names
        manager_employee_id          → manager_id       e.g. "ACQ_02436"; namespaced in clean.py
    """
    try:
        if not raw.get("employee_identifier"):
            return None

        name       = raw.get("name", {})
        contact    = raw.get("contact", {})
        assignment = raw.get("assignment", {})
        emp_type   = raw.get("employment", {}).get("type", "")

        return {
            "employee_id":    raw["employee_identifier"],
            "first_name":     name.get("first"),
            "last_name":      name.get("last"),
            "email":          contact.get("email"),
            "department":     assignment.get("department"),
            "job_title":      assignment.get("role"),
            "hire_date":      assignment.get("hire_timestamp"),
            "country":        assignment.get("location"),
            "employment_type": CONFIG["employment_type_map"].get(emp_type, emp_type),
            "manager_id":     raw.get("manager_employee_id"),
        }
    except (KeyError, TypeError, AttributeError):
        return None


def ingest_acquiredco_json(
    filepath: Path,
    page_size: int = CONFIG["api_page_size"],
) -> pd.DataFrame:
    """
    Load AcquiredCo employee data from JSON, simulating a paginated REST API.

    The full record list is sliced into pages of `page_size` records. Each page
    is processed and logged as a separate API call would be. Malformed records
    within a page are dead-lettered without halting the page.

    Parameters
    ----------
    filepath  : Path
        Path to acquiredco_api.json.
    page_size : int
        Records per simulated API page (default: CONFIG["api_page_size"] = 500).

    Returns
    -------
    pd.DataFrame
        Flattened, renamed employee records ready for schema alignment.
    """
    if not filepath.exists():
        _log_dead_letter("acquiredco_api", f"File not found: {filepath}")
        return pd.DataFrame(columns=STANDARD_EMPLOYEE_SCHEMA)

    try:
        payload = json.loads(filepath.read_text(encoding="utf-8"))
        all_records: list = payload["employees"]
    except Exception as exc:
        _log_dead_letter("acquiredco_api", f"Failed to parse JSON: {exc}")
        return pd.DataFrame(columns=STANDARD_EMPLOYEE_SCHEMA)

    total   = len(all_records)
    n_pages = (total + page_size - 1) // page_size

    logger.info(
        f"  [acquiredco_api] Paginated ingest started — "
        f"{total:,} records across {n_pages} pages (page_size={page_size})"
    )

    parsed_rows: list = []

    for page in range(n_pages):
        page_slice = all_records[page * page_size : (page + 1) * page_size]
        page_rows  = []

        for raw in page_slice:
            row = _parse_acquiredco_record(raw)
            if row is None:
                _log_dead_letter(
                    "acquiredco_api",
                    "Malformed or missing required fields",
                    raw,
                )
            else:
                page_rows.append(row)

        parsed_rows.extend(page_rows)
        logger.info(
            f"  [acquiredco_api] Page {page + 1}/{n_pages} — "
            f"{len(page_rows)} records ingested"
        )

    df = pd.DataFrame(parsed_rows)
    df["source_system"]  = "acquiredco_api"
    df["company_origin"] = "AcquiredCo"

    logger.info(f"  [acquiredco_api] {len(df):,} records  ←  {filepath.name}")
    return df


# ── Source 3: Benefits Enrollment (XML) ───────────────────────────────────────

def _xml_text(element: ET.Element, tag: str) -> str | None:
    """Return stripped text of a child tag, or None if absent or empty."""
    child = element.find(tag)
    return child.text.strip() if (child is not None and child.text) else None


def ingest_benefits_xml(filepath: Path) -> pd.DataFrame:
    """
    Parse the benefits enrollment XML using Python's built-in ElementTree.

    Parameters
    ----------
    filepath : Path
        Path to benefits_enrollment.xml.

    Returns
    -------
    pd.DataFrame
        Flat enrollment records with columns:
        employee_id, plan_type, coverage_level, enrollment_date,
        premium_employee, premium_employer.

    Note: employee_id is the raw numeric GlobalTech ID at this stage.
    Namespacing is not applied here — this DF is joined with the employee
    dataset by the visualization layer for the benefits enrollment rate chart.
    """
    if not filepath.exists():
        _log_dead_letter("benefits_xml", f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as exc:
        _log_dead_letter("benefits_xml", f"XML parse error: {exc}")
        return pd.DataFrame()

    rows: list = []
    for enrollment in root.findall("enrollment"):
        try:
            row = {
                "employee_id":      _xml_text(enrollment, "employee_id"),
                "plan_type":        _xml_text(enrollment, "plan_type"),
                "coverage_level":   _xml_text(enrollment, "coverage_level"),
                "enrollment_date":  _xml_text(enrollment, "enrollment_date"),
                "premium_employee": float(_xml_text(enrollment, "premium_employee") or 0),
                "premium_employer": float(_xml_text(enrollment, "premium_employer") or 0),
            }
            rows.append(row)
        except (ValueError, TypeError) as exc:
            _log_dead_letter(
                "benefits_xml",
                f"Malformed enrollment element: {exc}",
                ET.tostring(enrollment, encoding="unicode"),
            )

    df = pd.DataFrame(rows)
    logger.info(f"  [benefits_xml] {len(df):,} enrollment records  ←  {filepath.name}")
    return df


# ── Source 4: Payroll (Excel) ─────────────────────────────────────────────────

def ingest_payroll_xlsx(filepath: Path) -> pd.DataFrame:
    """
    Load the combined payroll Excel file into a DataFrame.

    Parameters
    ----------
    filepath : Path
        Path to payroll_data.xlsx.

    Returns
    -------
    pd.DataFrame
        Payroll records with columns:
        employee_id, source, base_salary, currency, pay_frequency,
        bonus_target_pct, effective_date.

    Note: base_salary is kept as raw string because some values contain
    currency symbols (e.g. "$85,000"). Symbol stripping and USD conversion
    happen in clean.py.
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


# ── Schema alignment ──────────────────────────────────────────────────────────

def align_to_employee_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a source DataFrame contains all standard employee schema columns.

    Missing columns are added as NaN. Source-specific extra columns are retained
    after the standard columns (not dropped) for downstream traceability.

    Applies to: globaltech_hris and acquiredco_api DataFrames.
    Does NOT apply to: benefits_xml or payroll_xlsx (domain-specific schemas).
    """
    for col in STANDARD_EMPLOYEE_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    extras = [c for c in df.columns if c not in STANDARD_EMPLOYEE_SCHEMA]
    return df[STANDARD_EMPLOYEE_SCHEMA + extras].copy()


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest_all_sources() -> dict:
    """
    Ingest all 4 data sources and return a dict of DataFrames.

    Returns
    -------
    dict with keys:
        "employees" — HRIS + AcquiredCo concatenated, aligned to standard schema
        "payroll"   — Payroll records (merged with employees at dedup stage)
        "benefits"  — Benefits enrollment records (joined for analytics/charts)
    """
    logger.info("── Ingestion: GlobalTech HRIS (CSV) ──────────────────────────")
    hris_df = ingest_globaltech_hris(CONFIG["files"]["globaltech_hris"])

    logger.info("── Ingestion: AcquiredCo API (JSON, paginated) ────────────────")
    acquiredco_df = ingest_acquiredco_json(CONFIG["files"]["acquiredco_api"])

    logger.info("── Ingestion: Benefits Enrollment (XML) ───────────────────────")
    benefits_df = ingest_benefits_xml(CONFIG["files"]["benefits_xml"])

    logger.info("── Ingestion: Payroll (Excel) ─────────────────────────────────")
    payroll_df = ingest_payroll_xlsx(CONFIG["files"]["payroll_xlsx"])

    # Align employee sources to the standard schema, then concatenate
    hris_aligned       = align_to_employee_schema(hris_df)
    acquiredco_aligned = align_to_employee_schema(acquiredco_df)
    employees_df       = pd.concat([hris_aligned, acquiredco_aligned], ignore_index=True)

    # Per-source record count summary
    logger.info("── Ingestion summary ──────────────────────────────────────────")
    for origin, count in employees_df["company_origin"].value_counts().items():
        logger.info(f"  {origin:20s}: {count:,} employee records")
    logger.info(f"  {'Total employees':20s}: {len(employees_df):,}")
    logger.info(f"  {'Payroll records':20s}: {len(payroll_df):,}")
    logger.info(f"  {'Benefits records':20s}: {len(benefits_df):,}")

    dead = get_dead_letters()
    if len(dead) > 0:
        logger.warning(f"  Dead-letter total    : {len(dead)} skipped records")

    print(employees_df)
    print("========================================================")
    print(payroll_df)
    print("========================================================")
    print(benefits_df)

    return {
        "employees": employees_df,
        "payroll":   payroll_df,
        "benefits":  benefits_df,
    }
