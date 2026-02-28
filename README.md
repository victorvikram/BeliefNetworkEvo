# Belief Networks Evolution

This project analyzes belief networks using data from the General Social Survey (GSS). It processes GSS data to explore correlations between beliefs, attitudes, and social factors, generating network visualizations that reveal how different beliefs and social attitudes interconnect.

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
BeliefNetworkEvo/
├── src/                    # Main source code
│   ├── analyzers/          # Network analysis (frustration, centrality, graph similarity, etc.)
│   ├── generators/         # Network generation (correlation matrices, conditioned networks)
│   ├── loaders/            # Data loading and cleaning (GSS import, cleaning pipeline)
│   └── visualizers/        # Network visualization (pyvis, temporal, static)
├── tests/                  # Test suite (pytest)
├── notebooks/              # Jupyter notebooks (tutorials + results)
├── data/                   # Data directory (see data/README.md)
│   ├── raw/                # Raw GSS data (gitignored)
│   ├── cache/              # Cached cleaned data (gitignored)
│   └── validation_plots/   # Data quality visualizations
├── docs/                   # Documentation (GSS codebook)
├── outputs/                # Generated outputs (gitignored)
├── archive/                # Legacy codebase (gitignored)
├── pyproject.toml          # Project configuration and dependencies
└── pytest.ini              # Test configuration
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -e .
   # Or for development:
   pip install -e ".[dev]"
   ```

2. **Get Data**
   - Download [GSS data](https://gss.norc.org/content/dam/gss/get-the-data/documents/sas/GSS_sas.zip)
   - Extract and place `gss7222_r4.sas7bdat` in `data/raw/`

3. **Run Tests**
   ```bash
   python -m pytest tests/ -v -m "not slow"
   ```

4. **Explore Notebooks**
   - `notebooks/lesson_0_creating_a_belief_network.ipynb` — Tutorial: create a belief network from scratch
   - `notebooks/constructing_frustration_measure.ipynb` — Constructing the frustration measure
   - `notebooks/edge_frustration_and_change.ipynb` — Edge frustration and belief change analysis
   - `notebooks/prog_cons_comparison.ipynb` — Progressive vs. conservative belief network comparison

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

## References

- GSS Data: [NORC at the University of Chicago](https://gss.norc.org/)
- Variable Documentation: [GSS Variables](https://gss.norc.org/documents/codebook/GSS_Codebook.pdf)
