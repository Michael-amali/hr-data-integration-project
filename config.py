from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

CONFIG = {
    # Directories
    "input_dir": RAW_DIR,
    "output_dir": PROCESSED_DIR,
    "logs_dir": LOGS_DIR,

    # Standard employee schema
    "employee_schema": [
        "employee_id", "first_name", "last_name", "email",
        "department", "job_title", "hire_date", "country",
        "employment_type", "manager_id", "source_system", "company_origin",
    ],

    # Employment type normalization
    "employment_type_map": {
        "FT":         "Full-Time",
        "PT":         "Part-Time",
        "CONTRACTOR": "Contractor",
        "Full-Time":  "Full-Time",
        "Part-Time":  "Part-Time",
        "Contractor": "Contractor",
    },

    "fx_rates": {
        "USD": 1.00,
        "EUR": 1.08,
        "GBP": 1.27,
    },

    "pay_frequency_multipliers": {
        "Annual":    1,
        "Monthly":   12,
        "Bi-Weekly": 26,
    },

    "id_prefix": {
        "GlobalTech": "GT",
        "AcquiredCo": "AC",
    },

    # Deduplication source priority
    "source_priority": {
        "globaltech_hris": 1,
        "acquiredco_api":  2,
        "payroll":         3,
        "benefits":        4,
    },

    # Simulated API pagination
    "api_page_size": 500,

    # Fuzzy matching (Pass 3 deduplication)
    "fuzzy_threshold":       88,
    "hire_date_window_days": 30,

    # Data quality validation
    "email_regex":       r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "employee_id_regex": r"^(GT|AC)-\d{6}$",
    "quality_threshold": 0.95,
    "valid_employment_types": ["Full-Time", "Part-Time", "Contractor"],
    "valid_currencies":       ["USD", "EUR", "GBP"],
    "salary_min_usd":  15_000,
    "salary_max_usd": 2_000_000,
    "hire_date_min":  "1970-01-01",
}

for _d in [CONFIG["input_dir"], CONFIG["output_dir"], CONFIG["logs_dir"]]:
    _d.mkdir(parents=True, exist_ok=True)
