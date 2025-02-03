# CLEAN - GSS Data Analysis Project

This project analyzes belief networks using data from the General Social Survey (GSS). It processes GSS data to explore correlations between beliefs, attitudes, and social factors, generating network visualizations that reveal how different beliefs and social attitudes interconnect.

## Repository Organization

This repository contains two main directories:
- `CLEAN/`: The current, actively maintained codebase with improved structure and documentation
- `MESSY/`: The original codebase that is being gradually migrated to CLEAN

We are in the process of migrating functionality from MESSY to CLEAN while improving code quality, documentation, and testing. All new development should happen in the CLEAN directory.

## Project Purpose

The main goal is to analyze and visualize belief networks by:
1. Identifying correlations between beliefs and social attitudes
2. Generating network visualizations where:
   - Nodes represent beliefs/attitudes
   - Edges represent significant correlations
   - Edge weights indicate correlation strength
3. Analyzing network properties to understand belief clustering and relationships

## Project Structure

```
CLEAN/
├── datasets/                           # Data processing pipeline
│   ├── raw_data/                      # Original GSS data
│   ├── cleaned_data/                  # Processed datasets
│   ├── cached_data/                   # Intermediate processing results
│   ├── validation_plots/              # Data quality visualizations
│   ├── import_gss.py                  # Initial data import
│   ├── prepare_cleaned_datasets.py    # Data cleaning orchestration
│   ├── clean_data.py                  # Core cleaning functions
│   └── validate_cleaned_datasets.py   # Validation checks
│
├── source_code/                        # Main analysis code
│   ├── analysis/                       # Analysis scripts
│   │   └── corr_make_network.py       # Main network generation script
│   ├── visualization/                  # Plotting utilities
│   └── generators/                     # Data generation tools
│
└── notebooks/                          # Analysis notebooks
    └── network_analysis.ipynb          # Interactive network exploration
```

## Quick Start

1. **Install Conda**
   - Download and install [Miniconda](https://docs.anaconda.com/miniconda/install/#)
   - Run this code to install Miniconda and add it to PATH:
      ```bash
      curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
      start /wait "" .\miniconda.exe /S /AddToPath=1
      del .\miniconda.exe
      ```
   - Close and reopen your terminal/command prompt

2. **Clone and Setup Environment**
   ```bash
   # Clone repository
   git clone [repository-url]

   # Create and activate conda environment
   conda env create -f pythons_beliefs.yml
   conda activate pythons_beliefs
   ```


3. **Get Data**
   - Download [GSS data](https://gss.norc.org/content/dam/gss/get-the-data/documents/sas/GSS_sas.zip)
   - Extract and place `gss7222_r4.sas7bdat` in `CLEAN/datasets/raw_data/`

4. **Process Data**
   ```bash
   # Import and clean data
   python CLEAN/datasets/import_gss.py
   python CLEAN/datasets/prepare_cleaned_datasets.py
   ```

5. **Generate Networks**
   ```bash
   python CLEAN/source_code/analysis/corr_make_network.py
   ```

## Data Cleaning Details

### Variable Transformations

The project handles several types of variables:
- Binary variables (Yes/No responses)
- Opinion scales (e.g., 1-7 agreement scales)
- Frequency measures (e.g., 0-8 attendance scales)
- Confidence measures (1-3 scales)
- Political variables (e.g., PARTYID, POLVIEWS)
- Voting records (VOTE__, PRES__)

### Data Variants

Some GSS variables have multiple versions across years. The cleaning process:
1. Identifies variant pairs (e.g., NATSPAC and NATSPACY)
2. Combines variants while preserving temporal information
3. Standardizes coding schemes across years

### Quality Checks

The validation process includes:
- Missing value analysis
- Distribution plots
- Correlation matrices
- Data type verification
- Range checks for numerical variables
- Category consistency checks
- Year span analysis
- Dataset comparison reports

## Output Files

The pipeline generates several outputs:
1. **Cleaned Datasets**
   - `cleaned_data_1.pkl`: Regular version
   - `cleaned_data_2.pkl`: Median-centered version

2. **Validation Results**
   - Data completeness visualizations
   - Distribution comparisons
   - Year coverage analysis
   - Data quality reports

3. **Network Analysis**
   - Network visualizations
   - Correlation matrices
   - Network metrics (centrality, clustering)
   - Interactive visualizations (via notebooks)

## Contributing

When adding new variables or modifying transformations:
1. Update the variable mappings in `clean_data.py`
2. Add appropriate validation checks
3. Run the full validation suite
4. Document any special handling requirements

## References

- GSS Data: [NORC at the University of Chicago](https://gss.norc.org/)
- Variable Documentation: [GSS Variables](https://gss.norc.org/documents/codebook/GSS_Codebook.pdf)
