from datetime import datetime
from pathlib import Path

import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CONFIG
from utils import logger


# Golden dataset

def export_golden_parquet(golden: pd.DataFrame, output_dir: Path) -> Path:

    parquet_dir = output_dir / "golden_employees.parquet"

    # Remove any stale partition files from previous runs before writing
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)

    table = pa.Table.from_pandas(golden, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(parquet_dir),
        partition_cols=["company_origin"],
        compression="snappy",
    )

    size_kb = sum(f.stat().st_size for f in parquet_dir.rglob("*.parquet")) / 1024
    logger.info(f"  Golden Parquet     : {parquet_dir}  ({size_kb:,.0f} KB, partitioned by company_origin)")
    return parquet_dir


def export_golden_csv(golden: pd.DataFrame, output_dir: Path) -> Path:
    """Write the golden employee dataset as a flat CSV."""
    path = output_dir / "golden_employees.csv"
    golden.to_csv(path, index=False, encoding="utf-8-sig")
    size_kb = path.stat().st_size / 1024
    logger.info(f"  Golden CSV         : {path}  ({size_kb:,.0f} KB, {len(golden):,} rows)")
    return path


# Ghost employees

def export_ghost_employees(ghosts: pd.DataFrame, output_dir: Path) -> Path:
    """
    Write the ghost employee report.
    """
    path = output_dir / "ghost_employees.csv"
    ghosts.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  Ghost employees    : {path}  ({len(ghosts):,} records)")
    return path


# Probable matches

def export_probable_matches(probable_matches: pd.DataFrame, output_dir: Path) -> Path:
    """
    Write the fuzzy-match pairs for HR review.
    """
    path = output_dir / "probable_matches_review.csv"
    probable_matches.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  Probable matches   : {path}  ({len(probable_matches):,} pairs)")
    return path


# Schema documentation

_SCHEMA_DOCS = [
    ("employee_id",            "str",       "Namespaced employee identifier",                  "GT-001042 / AC-001042"),
    ("first_name",             "str",       "Employee first name (NFC-normalized, title case)", "Michael"),
    ("last_name",              "str",       "Employee last name (NFC-normalized, title case)",  "King"),
    ("email",                  "str",       "Work email address (lowercase)",                   "michael.king@globaltech.com"),
    ("department",             "str",       "Organisational department (as-sourced)",           "Engineering"),
    ("job_title",              "str",       "Job title / role",                                "Data Analyst"),
    ("hire_date",              "datetime",  "Date employment started (timezone-naive UTC)",     "2016-09-21"),
    ("country",                "str",       "Country of employment",                            "Netherlands"),
    ("employment_type",        "str",       "Canonical employment type",                        "Full-Time / Part-Time / Contractor"),
    ("manager_id",             "str",       "Namespaced ID of the direct manager (nullable)",   "GT-012765"),
    ("source_system",          "str",       "Primary source system for this record",            "globaltech_hris"),
    ("company_origin",         "str",       "Company this employee originally belonged to",     "GlobalTech / AcquiredCo"),
    ("hire_date_out_of_range", "bool",      "True if hire_date is before 1970 or after today",  "False"),
    ("source_systems",         "str",       "All source systems this employee appears in",      "globaltech_hris,payroll"),
    ("dedup_method",           "str",       "Dedup pass that determined this record's status",  "single_source / exact_id / email_match"),
    ("base_salary",            "float",     "Base salary in original currency (numeric)",       "77935.0"),
    ("currency",               "str",       "Salary currency code",                             "EUR / GBP / USD"),
    ("pay_frequency",          "str",       "Pay frequency before annualisation",               "Monthly / Bi-Weekly / Annual"),
    ("bonus_target_pct",       "float",     "Target bonus as percentage of base salary",        "10.5"),
    ("effective_date",         "datetime",  "Date from which this payroll record is effective", "2023-05-12"),
    ("salary_usd_annual",      "float",     "Annualised salary in USD (base × freq × FX rate)", "1010037.60"),
    ("base_salary_original",   "str",       "Raw base_salary before symbol stripping",          "$70,315"),
    ("currency_original",      "str",       "Raw currency before cleaning",                     "USD"),
    ("pay_frequency_original", "str",       "Raw pay_frequency before cleaning",                "Monthly"),
]


def export_schema_documentation(output_dir: Path) -> Path:
    """Write a Markdown data dictionary for the golden employee dataset."""
    path = output_dir / "schema_documentation.md"
    lines = [
        "# Golden Employee Dataset — Schema Documentation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Partitioned by `company_origin` (GlobalTech / AcquiredCo) in the Parquet export.",
        "",
        "| Column | Data Type | Description | Example Value |",
        "|--------|-----------|-------------|---------------|",
    ]
    for col, dtype, desc, example in _SCHEMA_DOCS:
        lines.append(f"| `{col}` | {dtype} | {desc} | `{example}` |")

    lines += [
        "",
        "## Notes",
        "- `salary_usd_annual` = `base_salary` × pay_frequency_multiplier × fx_rate_to_usd",
        "- FX rates (fixed): USD=1.00, EUR=1.08, GBP=1.27",
        "- Pay frequency multipliers: Annual=1, Monthly=12, Bi-Weekly=26",
        "- Employee IDs that do not begin with `GT-` or `AC-` failed namespacing and were logged.",
        "- Department names are left as-sourced; no taxonomy normalization was applied.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  Schema docs        : {path}")
    return path


# Entry point

def export_all(result: dict, output_dir: Path) -> None:
    """
    Export all Deliverable 6 outputs.
    """
    logger.info("--- Export: Golden Dataset -------------------------------------")
    export_golden_parquet(result["golden"], output_dir)
    export_golden_csv(result["golden"], output_dir)

    logger.info("--- Export: Compliance Outputs ---------------------------------")
    export_ghost_employees(result["ghosts"], output_dir)
    export_probable_matches(result["probable_matches"], output_dir)

    logger.info("--- Export: Documentation --------------------------------------")
    export_schema_documentation(output_dir)
