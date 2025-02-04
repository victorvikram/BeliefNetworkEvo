"""
GSS Dataset Analysis Package
--------------------------
This package contains tools for loading, cleaning, and analyzing GSS survey data.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from .import_gss import import_dataset
from .clean_raw_data import clean_datasets, DataConfig

__all__ = ['import_dataset', 'clean_datasets', 'DataConfig', 'load_cleaned_datasets', 'prepare_and_cache_datasets']

# This can be empty or contain package-level imports/configuration 