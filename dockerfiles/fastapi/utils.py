import logging
import json
import pickle
import boto3
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

def load_model(model_name: str, alias: str) -> tuple:
    """
    Load a machine learning model from MLflow or a local file and load data dictionary from S3 or a local file.

    Parameters:
    - model_name (str): The name of the model to load from MLflow.
    - alias (str): The alias of the model version to load.

    Returns:
    - tuple: A tuple containing the loaded model, model version, and data dictionary.

    Raises:
    - FileNotFoundError: If local files for the model or data dictionary are not found.
    - MlflowException: If there is an error loading the model from MLflow.
    - Exception: For any other errors.
    """
    # Initialize logging for the function
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading model '{model_name}' with alias '{alias}' from MLflow")
        # Set the MLflow tracking URI to connect to the MLflow server
        mlflow.set_tracking_uri('http://mlflow:5000')
        # Initialize the MLflow client
        client_mlflow = mlflow.MlflowClient()
        
        # Retrieve model details using its alias from MLflow registry
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        # Load the model from MLflow registry source path
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        # Extract and convert the model version to an integer
        version_model_ml = int(model_data_mlflow.version)
        logger.info(f"Model '{model_name}' loaded from MLflow registry with version {version_model_ml}")
        
    except MlflowException as e:
        try:
            # Open the local model file in read-binary mode
            with open('/app/files/rf_model.pkl', 'rb') as file_ml:
                # Load the model using pickle
                model_ml = pickle.load(file_ml)
            # Set the version to 0 as this is a local model
            version_model_ml = 0
            logger.info(f"Model '{model_name}' loaded from local file")
            
        except FileNotFoundError as fnf_error:
            logger.error(f"Local model file not found: {fnf_error}")
            raise FileNotFoundError(f"Local model file not found: {fnf_error}")
        
        except Exception as e:
            logger.error(f"Unexpected error loading local model: {e}")
            raise Exception(f"Unexpected error loading local model: {e}")

    try:
        logger.info("Loading data dictionary from S3")
        # Initialize the S3 client
        s3 = boto3.client('s3')
        # Retrieve the data dictionary JSON file from S3
        result_s3 = s3.get_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
        # Read and decode the JSON content from the S3 file
        text_s3 = result_s3["Body"].read().decode()
        # Load the JSON data into a dictionary
        data_dictionary = json.loads(text_s3)
        # Convert standard scaler mean and std values to numpy arrays for processing
        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
        logger.info("Data dictionary loaded from S3")
        
    except Exception as e:
        try:
            # Open the local data dictionary file in read mode
            with open('/app/files/bike_sharing_demand_data_info.json', 'r') as file_s3:
                # Load the JSON data into a dictionary
                data_dictionary = json.load(file_s3)
            logger.info("Data dictionary loaded from local file")
            
        except FileNotFoundError as fnf_error:
            logger.error(f"Local data dictionary file not found: {fnf_error}")
            raise FileNotFoundError(f"Local data dictionary file not found: {fnf_error}")
        
        except Exception as e:
            logger.error(f"Unexpected error loading local data dictionary: {e}")
            raise Exception(f"Unexpected error loading local data dictionary: {e}")

    return model_ml, version_model_ml, data_dictionary


def one_hot_encode(df: pd.DataFrame, one_hot_cols: list) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to specified columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - one_hot_cols (list): List of column names to apply One-Hot Encoding to.

    Returns:
    - pd.DataFrame: The DataFrame with One-Hot Encoded columns.

    Raises:
    - KeyError: If any specified column is not found in the DataFrame.
    """
    # Initialize logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Applying One-Hot Encoding to columns: {one_hot_cols}")

        # Perform one-hot encoding on the specified columns
        df_encoded = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)

        logger.info("One-Hot Encoding applied successfully")
        return df_encoded
    
    except KeyError as e:
        logger.error(f"KeyError during One-Hot Encoding: {e}")
        raise KeyError(f"One or more specified columns not found in the DataFrame: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during One-Hot Encoding: {e}")
        raise Exception(f"Unexpected error during One-Hot Encoding: {e}")

def cyclic_encode(df: pd.DataFrame, columns: list, max_value: int = 23) -> pd.DataFrame:
    """
    Applies cyclic encoding to specified columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to apply Cyclic Encoding to.
    - max_value (int): The maximum value the cyclic variable can take (e.g., 23 for hours).

    Returns:
    - pd.DataFrame: The DataFrame with Cyclic Encoded columns added.

    Raises:
    - KeyError: If any specified column is not found in the DataFrame.
    """
    # Initialize logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Applying cyclic encoding to columns: {columns}")

        for col_name in columns:
            # Calculate the sine of the values in the column, scaled to [0, 2π]
            df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
            
            # Calculate the cosine of the values in the column, scaled to [0, 2π]
            df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
            
            # Drop the original column after encoding to prevent redundancy
            df = df.drop(col_name, axis=1)

        logger.info("Cyclic encoding applied successfully")
        return df
    
    except KeyError as e:
        logger.error(f"KeyError during Cyclic Encoding: {e}")
        raise KeyError(f"One or more specified columns not found in the DataFrame: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during Cyclic Encoding: {e}")
        raise Exception(f"Unexpected error during Cyclic Encoding: {e}")
