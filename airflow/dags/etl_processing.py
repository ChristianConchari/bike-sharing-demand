"""
This DAG extracts data from a source CSV file, transforms the data, and loads it into a destination S3 bucket
as two separate CSV files, one for training and one for testing. The split between training and testing data is 70/30.
"""
from datetime import timedelta
from airflow.decorators import dag, task
import pandas as pd
import awswrangler as wr

MARKDOWN_TEXT = """
# ETL Pipeline

This DAG extracts data from a source CSV file, transforms the data, and loads it into a destination S3 bucket
as two separate CSV files, one for training and one for testing. The split between training and testing data is 70/30.
"""

default_args = {
    'owner': 'Christian Conchari',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=15)
}

@dag(
    dag_id='process_etl_bike_sharing_data',
    description='ETL process for bike sharing demand data, splitting data into training and testing datasets',
    doc_md=MARKDOWN_TEXT,
    tags=['etl', 'bike_sharing_demand'],
    default_args=default_args,
    catchup=False
)
def etl_processing():
    """
    ETL process for bike sharing demand data, splitting data into training and testing datasets
    """
    @task()
    def get_data():
        """
        Load the original dataset from a CSV file and save it to an S3 bucket
        """
        # Obtain the original dataset
        df = pd.read_csv('data/train.csv')
        
        # Save the original dataset to S3
        data_path = 's3://mlflow/data/raw/bike_sharing_raw.csv'
        wr.s3.to_csv(df, data_path, index=False)
        
    @task.virtualenv()
    def feature_engineering():
        """
        Perform feature engineering on the dataset
        """
        # Get the absolute path of the src directory
        src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
        # Add the src directory to the Python path
        sys.path.append(src_path)

        # Now you can import your functions
        from encoding_functions import cyclic_encode, label_encode, one_hot_encode
        
        # Set paths for the original and processed data
        data_original_path = 's3://mlflow/data/raw/bike_sharing_raw.csv'
        data_processed_path = 's3://mlflow/data/processed/bike_sharing_processed.csv'
        
        # Read the original dataset from S3
        df = wr.s3.read_csv(data_original_path)
        
        # Drop duplicates and null values
        df = df.drop_duplicates()
        df = df.dropna()
        
        # Ensure the datetime column is in the correct format
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract year, month, weekday, weekend, and hour
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['hour'] = df['datetime'].dt.hour
        
        # Drop the datetime column
        df = df.drop(columns=['datetime'], axis=1)
        
        # Normalize count column by taking the log
        df['log_count'] = np.log(df['count'] + 1)
        df['log_casual'] = np.log(df['casual'] + 1)
        df['log_registered'] = np.log(df['registered'] + 1)
        
        # Drop the original count, casual, and registered columns
        df = df.drop(['count', 'casual', 'registered'], axis=1)
        
        # Calculate the correlation matrix
        corr = df.corr()
        
        # Find pairs of columns with high correlation
        high_corr_pairs = [
            # Pair of columns with correlation value
            (col1, col2) for col1 in corr.columns for col2 in corr.columns 
            # Ensure the pair is unique and the correlation value is greater than 0.85
            if col1 != col2 and abs(corr.loc[col1, col2]) > 0.85 
        ]
        
        # Identify columns to drop
        to_drop = set()

        for col1, col2 in high_corr_pairs:
            # Drop the column with the least correlation with the target
            if abs(corr.loc[col1, 'log_count']) > abs(corr.loc[col2, 'log_count']):
                to_drop.add(col2)
            else:
                to_drop.add(col1)

        # Drop the identified columns
        df = df.drop(columns=to_drop)

        # Encode using label encoding
        df = label_encode(df, ['holiday', 'workingday', 'year'])
        # Encode using one-hot encoding
        df = one_hot_encode(df, ['weather', 'month', 'weekday'])
        # Encode using cyclic encoding
        df = cyclic_encode(df, ['hour'])
        
        # Save the processed dataset to S3
        wr.s3.to_csv(df, data_processed_path, index=False)