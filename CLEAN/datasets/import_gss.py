import pyreadstat as prs
from pathlib import Path
import pickle
import os
from typing import Tuple, Dict
import pandas as pd

def import_dataset() -> Tuple[pd.DataFrame, Dict]:
    """
    Load GSS data from cache if available, otherwise load from source and cache.
    
    Returns:
        Tuple of (dataframe, metadata)
    """
    # Get the current file's directory
    data_dir = Path(__file__).parent
    cache_dir = data_dir / "cached_data"
    cache_file = cache_dir / "gss_cache.pkl"
    
    # Try to load from cache first
    if cache_file.exists():
        print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            df, meta = pickle.load(f)
        return df, meta
    
    # If no cache, load from source
    print("Loading from source file (this may take a while)...")
    file_path = data_dir / "raw_data" / "gss7222_r4.sas7bdat"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    raw_df, meta = prs.read_sas7bdat(str(file_path))
    
    # Cache the data
    cache_dir.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((raw_df, meta), f)
    print(f"Cached raw data to: {cache_file}")
    
    return raw_df, meta

if __name__ == "__main__":
    df, meta = import_dataset()
    print("Dataset loaded successfully")
