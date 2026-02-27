# Belief Network Evolution

This project analyzes belief networks using GSS (General Social Survey) data, focusing on correlation analysis and network visualization.

## Project Structure

```
BeliefNetworkEvo/CLEAN/
├── datasets/           # Data processing and cleaning modules
│   ├── raw_data/       # Place GSS .sas7bdat files here
│   └── cached_data/    # Auto-generated pickle caches
├── source_code/        # Core functionality
│   ├── loaders/        # Data import and cleaning
│   ├── generators/     # Network generation algorithms
│   ├── visualizers/    # Network visualization tools
│   ├── analyzers/      # Network analysis utilities
│   └── tests/          # Unit tests
├── notebooks/          # Jupyter notebooks for analysis
│   ├── tutorials/      # Getting started notebooks
│   └── results/        # Analysis result notebooks
└── requirements.txt    # Project dependencies
```

## Installation

### Prerequisites
- Python 3.9 or higher
- Conda (recommended for environment management)

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd BeliefNetworkEvo/CLEAN
```

2. Create and activate a new conda environment:
```bash
conda create -n pythons_beliefs python=3.9
conda activate pythons_beliefs
```

3. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

This will install all required dependencies automatically, including dev tools (pytest, black, flake8, jupyter).

### Data Setup

Download the GSS data file (`gss7222_r4.sas7bdat`) and place it in `datasets/raw_data/`. The first time you load data, it will be cached automatically.

### Verifying Installation

```python
from source_code.loaders.import_gss import import_dataset
from source_code.loaders.clean_raw_data import clean_datasets
from source_code.generators.corr_make_network import calculate_correlation_matrix
```

### Running Tests

```bash
cd CLEAN
python -m pytest source_code/tests/
```

## Dependencies

Core dependencies:
- pandas (>=2.0.0) - Data manipulation and analysis
- numpy (>=1.24.0) - Numerical computations
- networkx (>=3.1) - Network analysis and manipulation
- matplotlib (>=3.7.0) - Plotting and visualization
- seaborn (>=0.12.0) - Statistical data visualization
- scipy (>=1.9.0) - Scientific computing
- scikit-learn (>=1.3.0) - Graphical lasso regularization
- pyreadstat (>=1.2.0) - Reading SAS data files
- pingouin (>=0.5.3) - Partial correlations

## Usage

The project is organized into several main components:

1. **Data Processing** (`source_code/loaders/`)
   - `import_gss.py`: Import and cache GSS survey data
   - `clean_raw_data.py`: Data cleaning and normalization

2. **Network Generation** (`source_code/generators/`)
   - `corr_make_network.py`: Correlation network generation
   - `corr_make_conditioned_network.py`: Conditioned correlation networks

3. **Visualization** (`source_code/visualizers/`)
   - `network_visualizer.py`: Static network visualization
   - `temporal_network_visualizer.py`: Temporal network evolution

4. **Analysis** (`source_code/analyzers/`)
   - `graph_similarity.py`: Graph comparison utilities
   - `frustration_analyzer.py`: Belief frustration analysis
   - `centrality_analyzer.py`: Node centrality measures

Example usage can be found in the `notebooks/tutorials/` directory.
