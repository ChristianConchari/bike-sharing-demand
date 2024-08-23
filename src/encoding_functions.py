import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def label_encode(df, columns):
    """
    Applies Label Encoding to specified columns of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to apply Label Encoding to.
    
    Returns:
    pd.DataFrame: The DataFrame with the Label Encoded columns.
    """
    le = LabelEncoder()
    
    for col_label in columns:
        # Apply Label Encoding
        df[f'{col_label}_label'] = le.fit_transform(df[col_label])
    
    # Drop the original columns
    df = df.drop(columns, axis=1)
    
    return df

def one_hot_encode(df, one_hot_cols):
    """
    Applies One-Hot Encoding to specified columns of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    one_hot_cols (list): List of column names to apply One-Hot Encoding to.
    
    Returns:
    pd.DataFrame: The DataFrame with One-Hot Encoded columns.
    """
    # Apply One-Hot Encoding
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
    return df

def cyclic_encode(df, columns, max_value=23):
    """
    Applies cyclic encoding to specified columns of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to apply Cyclic Encoding to.
    max_value (int): The maximum value the cyclic variable can take (e.g., 23 for hours).
    
    Returns:
    pd.DataFrame: The DataFrame with Cyclic Encoded columns added.
    """
    for col_name in columns:
        # Compute the sin and cos components
        df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
        df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
        # Drop the original column
        df = df.drop(col_name, axis=1)
        
    return df