from datetime import datetime
from config import CONFIG
from utils import logger
from ingest import ingest_all_sources
from clean import clean_all
from deduplicate import run_deduplication
from validate import run_quality_checks
from visualize import generate_eda_report
from export import export_all


def run_pipeline() -> None:
    start = datetime.now()
    logger.info("=" * 65)
    logger.info("  GLOBALTECH HR DATA INTEGRATION PIPELINE")
    logger.info(f"  Run started: {start.isoformat()}")
    logger.info("=" * 65)

    # Step 1: Ingest
    logger.info("STEP 1: Ingestion")
    raw = ingest_all_sources()

    # Step 2: Clean
    logger.info("STEP 2: Cleaning & Transformation")
    cleaned = clean_all(raw)

    # Step 3: Deduplicate
    logger.info("STEP 3: Deduplication")
    result = run_deduplication(cleaned)
    golden   = result["golden"]
    benefits = result["benefits"]

    # Step 4: Validate
    logger.info("STEP 4: Data Quality Validation")
    quality_report = run_quality_checks(golden)

    # Step 5: Visualize
    logger.info("STEP 5: EDA & Visualization")
    generate_eda_report(golden, benefits, quality_report, CONFIG["output_dir"])

    # Step 6: Export
    logger.info("STEP 6: Export")
    export_all(result, CONFIG["output_dir"])

    # Summary
    duration = (datetime.now() - start).total_seconds()
    n_pass   = int((quality_report["status"] == "PASS").sum())
    n_checks = len(quality_report)

    logger.info("=" * 65)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Input employee records  : {len(raw['employees']):,}")
    logger.info(f"  Input payroll records   : {len(raw['payroll']):,}")
    logger.info(f"  Golden employee records : {len(golden):,}")
    logger.info(f"  Ghost employees         : {len(result['ghosts']):,}")
    logger.info(f"  Probable match pairs    : {len(result['probable_matches']):,}")
    logger.info(f"  Quality checks passed   : {n_pass}/{n_checks}")
    logger.info(f"  Duration                : {duration:.1f}s")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_pipeline()
