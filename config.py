"""Pipeline configuration: paths, constants, and all domain mappings."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

CONFIG = {
    # ── Directories ───────────────────────────────────────────────────────────
    "input_dir": RAW_DIR,
    "output_dir": PROCESSED_DIR,
    "logs_dir": LOGS_DIR,

    # ── Source file paths ─────────────────────────────────────────────────────
    "files": {
        "globaltech_hris": RAW_DIR / "globaltech_hris.csv",
        "acquiredco_api":  RAW_DIR / "acquiredco_api.json",
        "benefits_xml":    RAW_DIR / "benefits_enrollment.xml",
        "payroll_xlsx":    RAW_DIR / "payroll_data.xlsx",
    },

    # ── Standard employee schema ──────────────────────────────────────────────
    # These are the columns every employee record must have after ingest + align.
    # Benefits and Payroll are domain-specific DFs; they are NOT aligned to this
    # schema — payroll merges at the dedup stage, benefits is joined for analytics.
    "employee_schema": [
        "employee_id", "first_name", "last_name", "email",
        "department", "job_title", "hire_date", "country",
        "employment_type", "manager_id", "source_system", "company_origin",
    ],

    # ── Employment type normalization ─────────────────────────────────────────
    # AcquiredCo uses abbreviations; GlobalTech uses full names.
    # FT → Full-Time | PT → Part-Time | CONTRACTOR → Contractor
    "employment_type_map": {
        "FT":         "Full-Time",
        "PT":         "Part-Time",
        "CONTRACTOR": "Contractor",
        "Full-Time":  "Full-Time",
        "Part-Time":  "Part-Time",
        "Contractor": "Contractor",
    },

    # ── Currency → USD fixed exchange rates ───────────────────────────────────
    # Fixed snapshot rates (effective 2026-06-04).
    "fx_rates": {
        "USD": 1.00,
        "EUR": 1.08,
        "GBP": 1.27,
    },

    # ── Pay frequency → annual multiplier ────────────────────────────────────
    # Annual stays as-is | Monthly × 12 | Bi-Weekly × 26
    "pay_frequency_multipliers": {
        "Annual":    1,
        "Monthly":   12,
        "Bi-Weekly": 26,
    },

    # ── Employee ID namespace prefixes ────────────────────────────────────────
    # Both companies have overlapping raw numeric IDs; namespacing resolves this.
    # Format output: GT-XXXXXX (GlobalTech) | AC-XXXXXX (AcquiredCo)
    "id_prefix": {
        "GlobalTech": "GT",
        "AcquiredCo": "AC",
    },

    # ── Deduplication source priority (lower number = higher authority) ───────
    "source_priority": {
        "globaltech_hris": 1,
        "payroll":         2,
        "benefits":        3,
        "acquiredco_api":  4,
    },

    # ── Simulated API pagination ──────────────────────────────────────────────
    "api_page_size": 500,

    # ── Fuzzy matching (Pass 3 deduplication) ─────────────────────────────────
    "fuzzy_threshold":       88,   # minimum rapidfuzz similarity score (0–100)
    "hire_date_window_days": 30,   # hire dates must be within this many days to be compared

    # ── Data quality validation ───────────────────────────────────────────────
    "email_regex":       r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "employee_id_regex": r"^(GT|AC)-\d{6}$",
    "quality_threshold": 0.95,
    "valid_employment_types": ["Full-Time", "Part-Time", "Contractor"],
    "valid_currencies":       ["USD", "EUR", "GBP"],
    "salary_min_usd":  15_000,
    "salary_max_usd": 2_000_000,
    "hire_date_min":  "1970-01-01",

    # ── Department taxonomy ───────────────────────────────────────────────────
    # Decision: department names are left as-sourced; no normalization applied.
    # Rationale: preserves each company's org structure for post-merger analysis.
    # HR can align taxonomy in a follow-up data governance task.
    #
    # GlobalTech known departments:
    #   Manufacturing, Strategy, Human Resources, Marketing, Data Science,
    #   Product, Operations, DevOps, Sales, Business Development, Finance,
    #   Engineering, Legal
    # AcquiredCo known departments:
    #   Product, Legal, Marketing, Sales, Data Science, Human Resources,
    #   Finance, Engineering, Information Technology, Supply Chain,
    #   Business Development, Customer Success
}

for _d in [CONFIG["input_dir"], CONFIG["output_dir"], CONFIG["logs_dir"]]:
    _d.mkdir(parents=True, exist_ok=True)
