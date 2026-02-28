"""
Dataset Loading Package
--------------------------
This package contains tools for loading, cleaning, and analyzing two kinds of data:
- GSS raw data
- cleaned datasets
- outputed data usually correlation matrices

"""

from .import_gss import import_dataset
from .clean_raw_data import clean_datasets, DataConfig

__all__ = ['import_dataset', 'clean_datasets', 'DataConfig']