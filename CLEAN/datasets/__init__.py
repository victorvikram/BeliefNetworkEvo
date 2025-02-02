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

from .clean_data import clean_datasets
from .prepare_cleaned_datasets import load_cleaned_datasets, prepare_and_cache_datasets 