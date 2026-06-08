from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from utils import logger

PALETTE    = sns.color_palette("colorblind")
CLR_BLUE   = PALETTE[0]
CLR_ORANGE = PALETTE[1]
CLR_GREEN  = PALETTE[2]
CLR_RED    = PALETTE[3]
CLR_PURPLE = PALETTE[4]
CLR_PASS   = CLR_GREEN
CLR_FAIL   = CLR_RED


def _despine(ax):
    sns.despine(ax=ax, left=False, bottom=False)

def _annotate_source(ax, note: str):
    ax.text(
        0.99, 0.99, note,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7, color="#777777",
    )

# Chart 1: Headcount by Department

def chart_headcount_by_department(ax: plt.Axes, df: pd.DataFrame):
    counts = (
        df["department"]
        .value_counts()
        .head(18)
        .sort_values()
    )

    bars = ax.barh(counts.index, counts.values, color=CLR_GREEN, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Headcount", fontsize=10)
    ax.set_title("Headcount by Department", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _despine(ax)
    _annotate_source(ax, "Source: Unified employees from GlobalTech CSV, AcquireCo JSON")


# Chart 2: Headcount by Country

def chart_headcount_by_country(ax: plt.Axes, df: pd.DataFrame):
    counts = (
        df["country"]
        .value_counts()
        .head(20)
        .sort_values()
    )

    bars = ax.barh(counts.index, counts.values, color="#E69F00", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Headcount", fontsize=10)
    ax.set_title("Headcount by Country (Top 20)", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _despine(ax)
    _annotate_source(ax, "Source: Unified employees from GlobalTech CSV, AcquireCo JSON")


# Chart 3: Salary Distribution by Employment Type (Box Plot)

def chart_salary_by_employment_type(ax: plt.Axes, df: pd.DataFrame):
    salary_df = df.dropna(subset=["salary_usd_annual", "employment_type"]).copy()
    salary_df["salary_usd_k"] = salary_df["salary_usd_annual"] / 1_000

    order     = sorted(salary_df["employment_type"].unique())
    box_colors = {etype: PALETTE[i % len(PALETTE)] for i, etype in enumerate(order)}

    sns.boxplot(
        data=salary_df,
        x="employment_type",
        y="salary_usd_k",
        hue="employment_type",
        order=order,
        palette=box_colors,
        legend=False,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        ax=ax,
    )
    ax.set_xlabel("Employment Type", fontsize=10)
    ax.set_ylabel("Annual Salary (USD thousands)", fontsize=10)
    ax.set_title("Salary Distribution by Employment Type", fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
    _despine(ax)
    _annotate_source(ax, "Source: Unified employees from GlobalTech CSV, AcquireCo JSON, Payroll Excel")


# Chart 4: Tenure Distribution

def chart_tenure_distribution(ax: plt.Axes, df: pd.DataFrame):
    today     = pd.Timestamp.today()
    tenure_df = df.dropna(subset=["hire_date"]).copy()
    tenure_df["tenure_years"] = (today - tenure_df["hire_date"]).dt.days / 365.25
    tenure_df = tenure_df[tenure_df["tenure_years"] >= 0]

    ax.hist(
        tenure_df["tenure_years"],
        bins=30,
        color=CLR_BLUE,
        edgecolor="white",
        alpha=0.85,
    )

    ax.set_xlabel("Years of Tenure", fontsize=10)
    ax.set_ylabel("Number of Employees", fontsize=10)
    ax.set_title("Tenure Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _despine(ax)
    _annotate_source(ax, "Source: Unified employees from GlobalTech CSV, AcquireCo JSON")


# Chart 5: Benefits Enrollment Rate by Department 

def chart_benefits_enrollment_rate(
    ax: plt.Axes, golden: pd.DataFrame, benefits: pd.DataFrame
):
    """
    Enrollment rate = (GT employees enrolled in ≥1 benefit) / (all GT employees)
    per department.

    Only GlobalTech employees are considered; AcquiredCo uses a separate system.
    """
    gt = golden[golden["company_origin"] == "GlobalTech"].copy()

    # Map GT-XXXXXX → integer for matching against benefits employee_id
    gt["_id_int"] = (
        gt["employee_id"].str.replace("GT-", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    enrolled_ids = set(
        pd.to_numeric(benefits["employee_id"], errors="coerce")
        .dropna()
        .astype(int)
    )
    gt["enrolled"] = gt["_id_int"].isin(enrolled_ids)

    dept_stats = gt.groupby("department").agg(
        total_employees=("employee_id", "count"),
        enrolled_employees=("enrolled", "sum")
    )
    dept_stats["rate"] = (dept_stats["enrolled_employees"] / dept_stats["total_employees"])
    dept_stats = dept_stats.sort_values(by="rate").tail(20)

    bar_colors = [CLR_GREEN if r >= 0.5 else CLR_ORANGE for r in dept_stats["rate"]]
    bars = ax.barh(dept_stats.index, dept_stats["rate"], color=bar_colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{r:.0%}" for r in dept_stats["rate"]], padding=3, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Enrollment Rate", fontsize=10)
    ax.set_title("Benefits Enrollment Rate by Department\n(GlobalTech employees)", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)
    _despine(ax)
    _annotate_source(ax, "Source: Benefits enrollment XML")


# Chart 6: Data Quality Summary

def chart_data_quality_summary(ax: plt.Axes, quality_report: pd.DataFrame):
    """
    Grouped bar showing passed vs failed counts per quality check.
    """
    labels  = quality_report["check"]
    x       = np.arange(len(labels))
    width   = 0.38

    passed = quality_report["passed"].values
    failed = quality_report["failed"].values

    b1 = ax.bar(x - width / 2, passed, width, label="Passed", color=CLR_GREEN, alpha=0.85)
    b2 = ax.bar(x + width / 2, failed, width, label="Failed",  color=CLR_RED,   alpha=0.85)

    ax.bar_label(b1, fmt="%d", fontsize=6, padding=2)
    ax.bar_label(b2, fmt="%d", fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Record Count", fontsize=10)
    ax.set_title("Data Quality Summary — Passed vs Failed per Check",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _despine(ax)
    _annotate_source(ax, "Source: Unified employees from GlobalTech CSV, AcquireCo JSON, Payroll Excel")


# Entry point

def generate_eda_report(
    golden: pd.DataFrame,
    benefits: pd.DataFrame,
    quality_report: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Produce a 6-chart HR analytics report and save as a high-resolution PNG.
    """

    fig, axes = plt.subplots(3, 2, figsize=(26, 20))
    fig.suptitle(
        "GlobalTech HR Integration — Analytics",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    sns.set_style("whitegrid")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

    chart_headcount_by_department(axes[0, 0], golden)
    chart_headcount_by_country(axes[0, 1], golden)
    chart_salary_by_employment_type(axes[1, 0], golden)
    chart_tenure_distribution(axes[1, 1], golden)
    chart_benefits_enrollment_rate(axes[2, 0], golden, benefits)
    chart_data_quality_summary(axes[2, 1], quality_report)

    plt.tight_layout(rect=[0, 0, 1, 0.988])

    output_path = output_dir / "hr_integration_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  EDA report saved: {output_path}  ({size_kb:,.0f} KB, 300 DPI)")
