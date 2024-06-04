# data_loading.py

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pandas.DataFrame: DataFrame containing the loaded data.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    return df

if __name__ == "__main__":
    # Path to the CSV file
    file_path = 'sql_training_data.csv'
    
    # Load the data
    data_df = load_data(file_path)
    
    # Print the first few rows of the DataFrame
    print(data_df.head())
