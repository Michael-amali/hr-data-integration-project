import sys
from datetime import datetime
import pandas as pd
from config import CONFIG
from utils import logger


class DataQualityValidator:

    def __init__(self, df: pd.DataFrame, threshold: float = CONFIG["quality_threshold"]):
        self.df        = df
        self.threshold = threshold
        self.results   = []
        self.n         = len(df)

    # ── Internal recorder ─────────────────────────────────────────────────────

    def _record(self, check: str, description: str, failed: int, total: int) -> dict:
        pass_rate = 1 - (failed / total) if total > 0 else 1.0
        result = {
            "check":       check,
            "description": description,
            "total":       total,
            "passed":      total - failed,
            "failed":      failed,
            "pass_rate":   round(pass_rate, 4),
            "status":      "PASS" if pass_rate >= self.threshold else "FAIL",
        }
        self.results.append(result)
        return result

    # ── Check methods ─────────────────────────────────────────────────────────

    def check_not_null(self, column: str, description: str) -> dict:
        """Fail = number of null values in column."""
        failed = int(self.df[column].isna().sum())
        return self._record(f"NOT NULL: {column}", description, failed, self.n)

    def check_unique(self, column: str, description: str) -> dict:
        """Fail = number of duplicate non-null values."""
        non_null = self.df[column].dropna()
        failed   = int(non_null.duplicated().sum())
        return self._record(f"UNIQUE: {column}", description, failed, len(non_null))

    def check_values_in_set(self, column: str, valid_values: list, description: str) -> dict:
        """Fail = number of non-null values not in the allowed set."""
        non_null = self.df[column].dropna()
        failed   = int((~non_null.isin(valid_values)).sum())
        return self._record(f"VALUES IN SET: {column}", description, failed, len(non_null))

    def check_regex(self, column: str, pattern: str, description: str) -> dict:
        """Fail = number of non-null values that do not match the pattern."""
        non_null = self.df[column].dropna().astype(str)
        failed   = int((~non_null.str.match(pattern, na=False)).sum())
        return self._record(f"REGEX: {column}", description, failed, len(non_null))

    def check_numeric_range(
        self, column: str, min_val: float, max_val: float, description: str
    ) -> dict:
        """Fail = number of non-null values outside [min_val, max_val]."""
        non_null = pd.to_numeric(self.df[column], errors="coerce").dropna()
        failed   = int((~non_null.between(min_val, max_val)).sum())
        return self._record(f"NUMERIC RANGE: {column}", description, failed, len(non_null))

    def check_date_range(
        self, column: str, min_date: str, max_date: str, description: str
    ) -> dict:
        """Fail = number of non-null dates outside [min_date, max_date]."""
        non_null = pd.to_datetime(self.df[column], errors="coerce").dropna()
        in_range = non_null.between(pd.Timestamp(min_date), pd.Timestamp(max_date))
        failed   = int((~in_range).sum())
        return self._record(f"DATE RANGE: {column}", description, failed, len(non_null))

    def check_referential_integrity(
        self, fk_column: str, pk_column: str, description: str
    ) -> dict:
        """
        Fail = number of non-null fk_column values not present in pk_column.
        """
        fk_non_null   = self.df[fk_column].dropna().astype(str)
        fk_distinct   = set(fk_non_null)
        pk_values     = set(self.df[pk_column].dropna().astype(str))
        orphan_ids    = fk_distinct - pk_values
        failed        = len(orphan_ids)           # distinct bad FK values
        total         = len(fk_distinct) 
        return self._record(
            f"REFERENTIAL: {fk_column}->{pk_column}", description, failed, total
        )

    # ── Report ────────────────────────────────────────────────────────────────

    def generate_report(self) -> pd.DataFrame:
        """Compile results into a DataFrame and log a formatted summary."""
        report = pd.DataFrame(self.results)

        logger.info("=" * 65)
        logger.info("  DATA QUALITY REPORT")
        logger.info("=" * 65)
        for r in self.results:
            icon = "✓" if r["status"] == "PASS" else "✗"
            logger.info(
                f"  [{icon}] {r['check']:<42}  "
                f"{r['pass_rate']:>6.1%}  ({r['failed']} failed / {r['total']})"
            )

        n_fail = int((report["status"] == "FAIL").sum())
        overall = "ALL CHECKS PASSED ✓" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED ✗"
        logger.info("=" * 65)
        logger.info(f"  OVERALL: {overall}")
        logger.info("=" * 65)

        return report


# ── Pipeline gate ─────────────────────────────────────────────────────────────

def pipeline_gate(report: pd.DataFrame, max_failures: int = 2) -> None:
    """
    Halt the pipeline if too many checks have failed.

    Parameters
    ----------
    report       : quality report DataFrame
    max_failures : number of allowed FAIL checks before halting (default 2)

    Raises SystemExit if the failure count exceeds max_failures.
    """
    n_fail = int((report["status"] == "FAIL").sum())
    if n_fail > max_failures:
        logger.critical(
            f"PIPELINE GATE TRIGGERED: {n_fail} quality checks failed "
            f"(threshold: {max_failures}). Halting pipeline."
        )
        failed_checks = report.loc[report["status"] == "FAIL", "check"].tolist()
        logger.critical(f"Failed checks: {failed_checks}")
        sys.exit(1)
    elif n_fail > 0:
        logger.warning(
            f"Pipeline gate: {n_fail} check(s) failed but within tolerance "
            f"({max_failures} allowed). Pipeline continues."
        )


# ── HTML export ───────────────────────────────────────────────────────────────

_HTML_STYLE = """
    <style>
        body  { font-family: Arial, sans-serif; font-size: 13px; margin: 30px; }
        h1    { color: #333; }
        p     { color: #666; margin-top: 2px; }
        table { border-collapse: collapse; width: 100%; }
        th    { background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }
        td    { padding: 7px 12px; border-bottom: 1px solid #ddd; }
        tr:hover td { background: #f5f5f5; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
    </style>
    """

def export_html_report(report: pd.DataFrame, output_path) -> None:
    """Write a styled HTML quality report."""
    def _status_cell(val):
        cls = "pass" if val == "PASS" else "fail"
        return f'<span class="{cls}">{val}</span>'

    display = report.copy()
    display["pass_rate"] = display["pass_rate"].map(lambda x: f"{x:.1%}")
    display["status"]    = display["status"].map(_status_cell)

    table_html = display.to_html(index=False, escape=False, border=0)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>HR Data Quality Report</title>{_HTML_STYLE}</head>
        <body>
        <h1>GlobalTech HR Integration — Data Quality Report</h1>
        <p>Generated: {timestamp} &nbsp;|&nbsp; Total records: {report['total'].iloc[0]:,}</p>
        {table_html}
        </body>
        </html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"  HTML report exported: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 15 quality checks on the golden employee dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The golden dataset (post-dedup, post-payroll-merge).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    v = DataQualityValidator(df, threshold=CONFIG["quality_threshold"])

    # ── NOT NULL checks ───────────────────────────────────────────────────────
    v.check_not_null("employee_id", "Every employee must have an ID")
    v.check_not_null("first_name",  "Every employee must have a first name")
    v.check_not_null("last_name",   "Every employee must have a last name")
    v.check_not_null("email",       "Every employee must have an email address")
    v.check_not_null("department",  "Every employee must be assigned to a department")
    v.check_not_null("country",     "Every employee must have a country on record")

    # ── UNIQUE checks ─────────────────────────────────────────────────────────
    v.check_unique("email",       "Emails must be unique after deduplication")
    v.check_unique("employee_id", "Employee IDs must be unique after deduplication")

    # ── VALUES IN SET checks ──────────────────────────────────────────────────
    v.check_values_in_set(
        "employment_type",
        CONFIG["valid_employment_types"],
        "Employment type must be Full-Time, Part-Time, or Contractor",
    )
    v.check_values_in_set(
        "currency",
        CONFIG["valid_currencies"],
        "Currency must be USD, EUR, or GBP",
    )

    # ── REGEX checks ──────────────────────────────────────────────────────────
    v.check_regex(
        "email",
        CONFIG["email_regex"],
        "Email must match standard format",
    )
    v.check_regex(
        "employee_id",
        CONFIG["employee_id_regex"],
        "Employee ID must match GT-XXXXXX or AC-XXXXXX format",
    )

    # ── NUMERIC RANGE check ───────────────────────────────────────────────────
    v.check_numeric_range(
        "salary_usd_annual",
        CONFIG["salary_min_usd"],
        CONFIG["salary_max_usd"],
        f"Annual USD salary must be between ${CONFIG['salary_min_usd']:,} and ${CONFIG['salary_max_usd']:,}",
    )

    # ── DATE RANGE check ──────────────────────────────────────────────────────
    v.check_date_range(
        "hire_date",
        CONFIG["hire_date_min"],
        today,
        f"Hire date must be between {CONFIG['hire_date_min']} and today",
    )

    # ── REFERENTIAL INTEGRITY check ───────────────────────────────────────────
    v.check_referential_integrity(
        "manager_id",
        "employee_id",
        "Every non-null manager_id must exist as an employee_id in the dataset",
    )

    report = v.generate_report()

    # ── Export results ────────────────────────────────────────────────────────
    out_dir = CONFIG["output_dir"]

    csv_path = out_dir / "quality_report.csv"
    report.to_csv(csv_path, index=False)
    logger.info(f"  Quality report CSV : {csv_path}")

    html_path = out_dir / "quality_report.html"
    export_html_report(report, html_path)

    # ── Pipeline gate ─────────────────────────────────────────────────────────
    pipeline_gate(report, max_failures=2)

    return report
