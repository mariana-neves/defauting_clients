import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def consistent_dates(data, test=False):
    """
    Filter out rows where DATA_VENCIMENTO or DATA_PAGAMENTO
    are earlier than DATA_EMISSAO_DOCUMENTO.

    Parameters:
    data (DataFrame): The input DataFrame containing the columns 'DATA_VENCIMENTO', 'DATA_EMISSAO_DOCUMENTO', and 'DATA_PAGAMENTO'.

    Returns:
    DataFrame: A DataFrame containing only the rows with inconsistent dates.
    """
    if test == False:
        consistent_date = data[
            (data['DATA_VENCIMENTO'] > data['DATA_EMISSAO_DOCUMENTO']) |
            (data['DATA_PAGAMENTO'] > data['DATA_EMISSAO_DOCUMENTO'])
        ]
    else:
        consistent_date = data[
            (data['DATA_VENCIMENTO'] > data['DATA_EMISSAO_DOCUMENTO'])
        ]

    return consistent_date


def get_numeric_cols(df, exclude_column='ID_CLIENTE'):
    """
    Returns a list of numeric columns in the given dataframe, excluding a specified column.

    Parameters:
    df (pd.DataFrame): The dataframe to check.
    exclude_column (str): The column to exclude from the list of numeric columns.

    Returns:
    list: A list of numeric column names, excluding the specified column.
    """
    # Select numeric columns using pandas' select_dtypes method
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the specified column if it exists in the list
    if exclude_column in numeric_columns:
        numeric_columns.remove(exclude_column)
    
    return numeric_columns


def map_state_region(cep_prefix, state_region_info):
    """
    Maps the first two digits of the postal code (CEP) to the corresponding state and region.
    
    :param cep_prefix: The first two digits of the postal code (CEP)
    :param state_region_info: Dictionary with postal code, state, and region information
    :return: Tuple containing the state and region, or (np.nan, np.nan) if not found
    """
    # Check if cep_prefix is not NaN
    if pd.isna(cep_prefix):
        return np.nan, np.nan
    
    # Check if cep_prefix can be transformed to int
    try:
        cep_prefix_int = int(cep_prefix)
    except ValueError:
        return np.nan, np.nan
    
    for state, info in state_region_info.items():
        if cep_prefix_int in info['CEP']:
            return state, info['REGIAO'][0]
    
    return np.nan, np.nan  # Return (np.nan, np.nan) if the postal code is not found

def add_state_region(df, json_file_path):
    """
    Adds 'ESTADO' and 'REGIAO' columns to the DataFrame based on the postal code (CEP).
    
    :param df: DataFrame containing the 'CEP_2_DIG' column
    :param json_file_path: Path to the JSON file with state and region information
    :return: DataFrame with the 'ESTADO' and 'REGIAO' columns added
    """
    # Load the state and region information dictionary
    with open(json_file_path, 'r') as f:
        state_region_info = json.load(f)
    
    # Apply the mapping to the DataFrame
    df['ESTADO'], df['REGIAO'] = zip(*df['CEP_2_DIG'].apply(lambda x: map_state_region(x, state_region_info)))
    
    return df


# def dummys_KNN(df):  
#     encoders = {}
#     for col in df.columns:
#         print(col)
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         encoders[col] = le
#     return df, encoders

    
# def evaluate_knn_imputation(data, n_neighbors):
#     """
#     Evaluates KNN imputation with a specified number of neighbors.

#     Parameters:
#     data (DataFrame): The input DataFrame with missing values to be imputed.
#     n_neighbors (int): The number of neighbors to use for KNN imputation.

#     Returns:
#     float: The Mean Squared Error (MSE) of the imputation on the test set.
#     """
#     # Split the data into training and test sets
#     train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    
#     # Apply KNN Imputer on the training data
#     imputer = KNNImputer(n_neighbors=n_neighbors)
#     train_imputed = imputer.fit_transform(train_data)
#     test_imputed = imputer.transform(test_data)

#     # Calculate MSE on test set (for numeric columns)
#     mse = mean_squared_error(test_data, test_imputed)
#     return mse
