import pyreadstat as prs
from pathlib import Path
import pickle
import os
from typing import Tuple, Dict, Optional, List
import pandas as pd
import sys
import time
import threading
import queue

def cat_loading_animation(stop_event):
    """
    Display a cat animation until the stop event is set.
    
    Args:
        stop_event: Threading event to signal when to stop the animation
    """
    frames = [
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
    
    while not stop_event.is_set():
        for frame in frames:
            if stop_event.is_set():
                break
            sys.stdout.write(f"\rThis could take a while... {frame}")
            sys.stdout.flush()
            time.sleep(0.2)
        if stop_event.is_set():
            break
    
    sys.stdout.write("\rDataset loaded! 🐱✨       \n")

def import_dataset() -> Tuple[pd.DataFrame, Dict]:
    """
    Load GSS data from cache if available, otherwise load from source and cache.
    
    Args:
        columns: Optional list of column names to load. If None, loads all columns.
    
    Returns:
        Tuple of (dataframe, metadata)
    """
    columns_of_interest = ['PARTYID', 'POLVIEWS', 'NATSPAC', 'NATENVIR', 'NATHEAL', 'NATCITY', 'NATCRIME', 'NATDRUG', 
                           'NATEDUC', 'NATRACE', 'NATARMS', 'NATAID', 'NATFARE', 'NATROAD', 'NATSOC', 'NATMASS', 'NATPARK', 
                           'NATCHLD', 'NATSCI', 'NATENRGY', 'NATSPACY', 'NATENVIY', 'NATHEALY', 'NATCITYY', 'NATCRIMY', 
                           'NATDRUGY', 'NATEDUCY', 'NATRACEY', 'NATARMSY', 'NATAIDY', 'NATFAREY', 'EQWLTH', 'SPKATH', 
                           'COLATH', 'LIBATH', 'SPKRAC', 'COLRAC', 'LIBRAC', 'SPKCOM', 'COLCOM', 'LIBCOM', 'SPKMIL', 
                           'COLMIL', 'LIBMIL', 'SPKHOMO', 'COLHOMO', 'LIBHOMO', 'SPKMSLM', 'COLMSLM', 'LIBMSLM', 'CAPPUN', 
                           'GUNLAW', 'COURTS', 'GRASS', 'POSTLIFE', 'PRAYER', 'AFFRMACT', 'WRKWAYUP', 'HELPFUL', 
                           'FAIR', 'TRUST', 'CONFINAN', 'CONBUS', 'CONCLERG', 'CONEDUC', 'CONFED', 'CONLABOR', 'CONPRESS', 
                           'CONMEDIC', 'CONTV', 'CONJUDGE', 'CONSCI', 'CONLEGIS', 'CONARMY', 'OBEY', 'POPULAR', 'THNKSELF', 
                           'WORKHARD', 'HELPOTH', 'GETAHEAD', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 
                           'ABSINGLE', 'ABANY', 'SEXEDUC', 'DIVLAW', 'PREMARSX', 'TEENSEX', 'XMARSEX', 'HOMOSEX', 'PORNLAW', 
                           'SPANKING', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP', 
                           'POLATTAK', 'FECHLD', 'FEPRESCH', 'FEFAM', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4', 
                           'HELPPOOR', 'HELPNOT', 'HELPBLK', 'MARHOMO','VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 
                           'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 'VOTE80', 'PRES80', 'IF80WHO', 'VOTE84', 'PRES84', 
                           'IF84WHO', 'VOTE88', 'PRES88', 'IF88WHO', 'VOTE92', 'PRES92', 'IF92WHO', 'VOTE96', 'PRES96', 
                           'IF96WHO', 'VOTE00', 'PRES00', 'IF00WHO', 'VOTE04', 'PRES04', 'IF04WHO', 'VOTE08', 'PRES08', 
                           'IF08WHO', 'VOTE12', 'PRES12', 'IF12WHO', 'VOTE16', 'PRES16', 'IF16WHO', 'VOTE20', 'PRES20', 
                           'IF20WHO', 'RELIG', 'ATTEND', 'RACOPEN', 'NEWS', 'TVHOURS', 'RELITEN']
    columns = columns_of_interest + ['YEAR', 'ID', 'BALLOT'] 

    #columns = columns + ["SEXFREQ"]
    
    # Get the current file's directory
    data_dir = Path(__file__).parent
    cache_dir = data_dir / "cached_data"
    cache_file = cache_dir / "gss_cache.pkl"
    
    # Try to load from cache first
    if cache_file.exists():
        print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            df, meta = pickle.load(f)
            # Filter columns after loading if specified
            if columns is not None:
                df = df[columns]
        print("Done! ✨")
        return df, meta
        
    
    # If no cache, load from source
    # Start the loading animation in a separate thread
    stop_animation = threading.Event()
    animation_thread = threading.Thread(target=cat_loading_animation, args=(stop_animation,))
    animation_thread.start()

    try:
        file_path = data_dir / "raw_data" / "gss7222_r4.sas7bdat"
        
        #if not file_path.exists():
        #    stop_animation.set()
        #    animation_thread.join()
        #    raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        raw_df, meta = prs.read_sas7bdat(
            str(file_path),
            user_missing=True,
            disable_datetime_conversion=True,
            formats_as_category=False,
            usecols=columns  # Only read specified columns if provided
        )


        #raw_df, meta = prs.read_sas7bdat(str(file_path), usecols=columns) # Only read specified columns if provided

        # Cache the full data
        cache_dir.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((raw_df, meta), f)
        
        # Stop the animation
        stop_animation.set()
        animation_thread.join()
        print(f"Cached raw data to: {cache_file}")
        
        return raw_df, meta
        
    except Exception as e:
        # Make sure to stop the animation if there's an error
        stop_animation.set()
        animation_thread.join()
        raise e


columns_of_interest = ['PARTYID', 'POLVIEWS', 'NATSPAC', 'NATENVIR', 'NATHEAL', 'NATCITY', 'NATCRIME', 'NATDRUG', 
                        'NATEDUC', 'NATRACE', 'NATARMS', 'NATAID', 'NATFARE', 'NATROAD', 'NATSOC', 'NATMASS', 'NATPARK', 
                        'NATCHLD', 'NATSCI', 'NATENRGY', 'NATSPACY', 'NATENVIY', 'NATHEALY', 'NATCITYY', 'NATCRIMY', 
                        'NATDRUGY', 'NATEDUCY', 'NATRACEY', 'NATARMSY', 'NATAIDY', 'NATFAREY', 'EQWLTH', 'SPKATH', 
                        'COLATH', 'LIBATH', 'SPKRAC', 'COLRAC', 'LIBRAC', 'SPKCOM', 'COLCOM', 'LIBCOM', 'SPKMIL', 
                        'COLMIL', 'LIBMIL', 'SPKHOMO', 'COLHOMO', 'LIBHOMO', 'SPKMSLM', 'COLMSLM', 'LIBMSLM', 'CAPPUN', 
                        'GUNLAW', 'COURTS', 'GRASS', 'POSTLIFE', 'PRAYER', 'AFFRMACT', 'WRKWAYUP', 'HELPFUL', 
                        'FAIR', 'TRUST', 'CONFINAN', 'CONBUS', 'CONCLERG', 'CONEDUC', 'CONFED', 'CONLABOR', 'CONPRESS', 
                        'CONMEDIC', 'CONTV', 'CONJUDGE', 'CONSCI', 'CONLEGIS', 'CONARMY', 'OBEY', 'POPULAR', 'THNKSELF', 
                        'WORKHARD', 'HELPOTH', 'GETAHEAD', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 
                        'ABSINGLE', 'ABANY', 'SEXEDUC', 'DIVLAW', 'PREMARSX', 'TEENSEX', 'XMARSEX', 'HOMOSEX', 'PORNLAW', 
                        'SPANKING', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP', 
                        'POLATTAK', 'FECHLD', 'FEPRESCH', 'FEFAM', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4', 
                        'HELPPOOR', 'HELPNOT', 'HELPBLK', 'MARHOMO','VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 
                        'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 'VOTE80', 'PRES80', 'IF80WHO', 'VOTE84', 'PRES84', 
                        'IF84WHO', 'VOTE88', 'PRES88', 'IF88WHO', 'VOTE92', 'PRES92', 'IF92WHO', 'VOTE96', 'PRES96', 
                        'IF96WHO', 'VOTE00', 'PRES00', 'IF00WHO', 'VOTE04', 'PRES04', 'IF04WHO', 'VOTE08', 'PRES08', 
                        'IF08WHO', 'VOTE12', 'PRES12', 'IF12WHO', 'VOTE16', 'PRES16', 'IF16WHO', 'VOTE20', 'PRES20', 
                        'IF20WHO', 'RELIG', 'ATTEND', 'RACOPEN', 'NEWS', 'TVHOURS', 'RELITEN']
columns_of_interest = columns_of_interest + ['YEAR', 'ID', 'BALLOT']


#------------------------------------------------------------------------------
# FOR TESTING (if run as script)
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage with specific columns
    # These are the standard columns that we will use for the analysis. Adding more columns will slow down the loading process... be careful! 🐱
    columns_of_interest = ['PARTYID', 'POLVIEWS', 'NATSPAC', 'NATENVIR', 'NATHEAL', 'NATCITY', 'NATCRIME', 'NATDRUG', 
                           'NATEDUC', 'NATRACE', 'NATARMS', 'NATAID', 'NATFARE', 'NATROAD', 'NATSOC', 'NATMASS', 'NATPARK', 
                           'NATCHLD', 'NATSCI', 'NATENRGY', 'NATSPACY', 'NATENVIY', 'NATHEALY', 'NATCITYY', 'NATCRIMY', 
                           'NATDRUGY', 'NATEDUCY', 'NATRACEY', 'NATARMSY', 'NATAIDY', 'NATFAREY', 'EQWLTH', 'SPKATH', 
                           'COLATH', 'LIBATH', 'SPKRAC', 'COLRAC', 'LIBRAC', 'SPKCOM', 'COLCOM', 'LIBCOM', 'SPKMIL', 
                           'COLMIL', 'LIBMIL', 'SPKHOMO', 'COLHOMO', 'LIBHOMO', 'SPKMSLM', 'COLMSLM', 'LIBMSLM', 'CAPPUN', 
                           'GUNLAW', 'COURTS', 'GRASS', 'POSTLIFE', 'PRAYER', 'AFFRMACT', 'WRKWAYUP', 'HELPFUL', 
                           'FAIR', 'TRUST', 'CONFINAN', 'CONBUS', 'CONCLERG', 'CONEDUC', 'CONFED', 'CONLABOR', 'CONPRESS', 
                           'CONMEDIC', 'CONTV', 'CONJUDGE', 'CONSCI', 'CONLEGIS', 'CONARMY', 'OBEY', 'POPULAR', 'THNKSELF', 
                           'WORKHARD', 'HELPOTH', 'GETAHEAD', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 
                           'ABSINGLE', 'ABANY', 'SEXEDUC', 'DIVLAW', 'PREMARSX', 'TEENSEX', 'XMARSEX', 'HOMOSEX', 'PORNLAW', 
                           'SPANKING', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP', 
                           'POLATTAK', 'FECHLD', 'FEPRESCH', 'FEFAM', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4', 
                           'HELPPOOR', 'HELPNOT', 'HELPBLK', 'MARHOMO','VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 
                           'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 'VOTE80', 'PRES80', 'IF80WHO', 'VOTE84', 'PRES84', 
                           'IF84WHO', 'VOTE88', 'PRES88', 'IF88WHO', 'VOTE92', 'PRES92', 'IF92WHO', 'VOTE96', 'PRES96', 
                           'IF96WHO', 'VOTE00', 'PRES00', 'IF00WHO', 'VOTE04', 'PRES04', 'IF04WHO', 'VOTE08', 'PRES08', 
                           'IF08WHO', 'VOTE12', 'PRES12', 'IF12WHO', 'VOTE16', 'PRES16', 'IF16WHO', 'VOTE20', 'PRES20', 
                           'IF20WHO', 'RELIG', 'ATTEND', 'RACOPEN', 'NEWS', 'TVHOURS', 'RELITEN']
    columns_of_interest = columns_of_interest + ['YEAR', 'ID', 'BALLOT']
    
    df, meta = import_dataset()
    print("Dataset loaded successfully")