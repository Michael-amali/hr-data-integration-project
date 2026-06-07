# hr-data-integration-project

ETL skeleton: ingest → clean → deduplicate → validate → visualize → export.

## Project layout

```
hr-data-integration-project/
├── pipeline.py              # Entry point
├── config.py                # All configuration
├── ingest.py                # Source-specific ingestion
├── clean.py                 # Cleaning functions
├── deduplicate.py           # Duplicates removals
├── validate.py              # Data quality checks
├── visualize.py             # Data presentation
├── export.py                # Output writing
├── utils.py                 # Shared helpers
├── data/
│   ├── raw/                 # Never modify
│   └── processed/           # Pipeline outputs
├── logs/
├── requirements.txt
└── README.md
```

## Conventions

- **Never edit** files under `data/raw/` — treat it as immutable source data.
- Write pipeline outputs to `data/processed/`.
- Logs are written under `logs/` (log files are gitignored).

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (after implementation)

```powershell
python pipeline
```


## Save packages
```powershell
pip freeze > requirements.txt
```