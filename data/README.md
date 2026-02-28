# Data Directory

This directory holds GSS (General Social Survey) data used by the project.

## Setup

1. Download the GSS cumulative data file in SAS format (`gss7222_r4.sas7bdat`)
   from [NORC at the University of Chicago](https://gss.norc.org/).
2. Place it in `data/raw/gss7222_r4.sas7bdat`.

The first time you run `import_dataset()`, the raw file will be read and a
cache saved to `data/cache/` for faster subsequent loads.

## Directory Structure

- `raw/` — Raw GSS data file (gitignored)
- `cache/` — Cached/cleaned data (gitignored)
- `validation_plots/` — Data validation visualizations
