# Belief Network Evolution

This project analyzes belief networks using GSS (General Social Survey) data, focusing on correlation analysis and network visualization.

## Project Structure

```
BeliefNetworkEvo/CLEAN/
├── datasets/           # Data processing and cleaning modules
├── source_code/       # Core functionality
│   ├── generators/    # Network generation algorithms
│   ├── visualizers/   # Network visualization tools
│   ├── analyzers/     # Network analysis utilities
│   └── tests/         # Unit tests
├── notebooks/         # Jupyter notebooks for analysis
└── requirements.txt   # Project dependencies
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
pip install -e .
```

This will install all required dependencies automatically. For development work, the following packages will also be installed:
- pytest (testing)
- black (code formatting)
- flake8 (code linting)
- jupyter and notebook (for running analysis notebooks)

### Verifying Installation

To verify the installation, open a Python console or Jupyter notebook and try importing the main modules:

```python
from datasets.import_gss import import_dataset
from datasets.clean_raw_data import clean_datasets
from source_code.generators.corr_make_network import calculate_correlation_matrix
```

## Dependencies

Core dependencies:
- pandas (≥2.0.0) - Data manipulation and analysis
- numpy (≥1.24.0) - Numerical computations
- networkx (≥3.1) - Network analysis and manipulation
- matplotlib (≥3.7.0) - Plotting and visualization
- seaborn (≥0.12.0) - Statistical data visualization
- scipy (≥1.9.0) - Scientific computing
- scikit-learn (≥1.3.0) - Required for partial correlations
- sas7bdat (≥2.2.3) - Reading SAS data files

Development dependencies:
- jupyter (=1.0.0)
- notebook (=7.0.6)
- pytest (≥7.0.0)
- black (≥22.0.0)
- flake8 (≥5.0.0)

## Usage

The project is organized into several main components:

1. Data Processing (`datasets/`)
   - `import_gss.py`: Functions for importing GSS survey data
   - `clean_raw_data.py`: Data cleaning and preprocessing utilities

2. Belief Network Generation (`source_code/generators/`)
   - `corr_make_network.py`: Correlation network generation
   - `corr_make_conditioned_network.py`: Conditioned correlation network analysis

3. Visualisation (`source_code/visualizers/`)
   - `network_visualizer.py`: Network visualization tools

4. Analysis (`source_code/analyzers/`)
   - `graph_similarity.py`: Graph comparison utilities

Example usage can be found in the notebooks directory.
