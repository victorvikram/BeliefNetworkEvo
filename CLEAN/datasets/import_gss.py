import pyreadstat as prs
from pathlib import Path
import pickle
import os
from typing import Tuple, Dict
import pandas as pd
import sys
import time

def cat_loading_animation():
    """
    Ignore this.
    """

    frames = [
        "🐱  ",
        "🐾🐱 ",
        " 🐾🐱",
        "  🐾🐱",
        "   🐾🐱",
        "    🐾🐱",
        "   🐾🐱",
        "  🐾🐱",
        " 🐾🐱",
        "🐾🐱 "
    ]
    
    for _ in range(3):  # Repeat the animation a few times
        for frame in frames:
            sys.stdout.write(f"\rJust a moment... {frame} ")
            sys.stdout.flush()
            time.sleep(0.2)

    sys.stdout.write("\rDone! 🐱✨       \n")

# Call the function while your file is "loading"


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
    cat_loading_animation()

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
