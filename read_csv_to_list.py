import pandas as pd

def print_csv_as_list(csv_path):
    """
    Read a CSV file and print its column names as a Python list.
    
    Args:
        csv_path: Path to CSV file
    """
    try:
        # Read CSV with explicit encoding
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Get column names
        columns = df.columns.tolist()
        
        # Print formatted list on one line
        print("[", end="")
        for i, col in enumerate(columns):
            if i < len(columns) - 1:
                print(f"'{col}', ", end="")
            else:
                print(f"'{col}'", end="")
        print("]")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_path}'")
        print("Please check if the file exists and the path is correct")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    csv_path = "symm_vars.csv"
    print_csv_as_list(csv_path) 