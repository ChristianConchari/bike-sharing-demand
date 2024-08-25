import json
import pickle
import boto3
import mlflow
import joblib

import numpy as np
import pandas as pd
import os

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

# Define global variables
model_name = "bike_sharing_model_prod"
version_model = 0
data_dictionary = {}

def load_model(model_name: str, alias: str):
    global model, version_model, data_dictionary
    
    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()
        
        # Get the model version from the MLflow registry
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        # Load the model
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        # Get the version of the model
        version_model_ml = int(model_data_mlflow.version)
        
        print(f"Model {model_name} loaded from MLflow registry with version {version_model_ml}")
        
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/rf_model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0
        
        print(f"Model {model_name} loaded from local file")
        
    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/bike_sharing_demand_data_info.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
        
        print("Data dictionary loaded from S3")
        
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/bike_sharing_demand_data_info.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()
        
        print("Data dictionary loaded from local file")

    return model_ml, version_model_ml, data_dictionary

class ModelInput(BaseModel):
    """
    Input schema for the heart disease prediction model.

    This class defines the input fields required by the heart disease prediction model along with their descriptions
    and validation constraints.

    :param age: Age of the patient (0 to 150).
    :param sex: Sex of the patient. 1: male; 0: female.
    :param cp: Chest pain type. 1: typical angina; 2: atypical angina; 3: non-anginal pain; 4: asymptomatic.
    :param trestbps: Resting blood pressure in mm Hg on admission to the hospital (90 to 220).
    :param chol: Serum cholestoral in mg/dl (110 to 600).
    :param fbs: Fasting blood sugar. 1: >120 mg/dl; 0: <120 mg/dl.
    :param restecg: Resting electrocardiographic results. 0: normal; 1: having ST-T wave abnormality; 2: showing
                    probable or definite left ventricular hypertrophy.
    :param thalach: Maximum heart rate achieved (beats per minute) (50 to 210).
    :param exang: Exercise induced angina. 1: yes; 0: no.
    :param oldpeak: ST depression induced by exercise relative to rest (0.0 to 7.0).
    :param slope: The slope of the peak exercise ST segment. 1: upsloping; 2: flat; 3: downsloping.
    :param ca: Number of major vessels colored by flourosopy (0 to 3).
    :param thal: Thalassemia disease. 3: normal; 6: fixed defect; 7: reversable defect.
    """
    season: Literal[1, 2, 3, 4] = Field(
        description="The season of the year. 1: winter; 2: spring; 3: summer; 4: fall"
    )
    yr: int = Field(
        description="The year (0: 2011, 1: 2012)",
        ge=0,
        le=1,
    )
    hr: int = Field(
        description="The hour of the day (0 to 23)",
        ge=0,
        le=23,
    )
    holiday: Literal[0, 1] = Field(
        description="Whether the day is a holiday or not. 1: holiday; 0: no holiday"
    )
    weekday: int = Field(
        description="The day of the week",
        ge=0,
        le=6,
    )
    workingday: Literal[0, 1] = Field(
        description="Whether the day is a working day or not. 1: working day; 0: no working day"
    )
    weathersit: Literal[1, 2, 3, 4] = Field(
        description="Clear, Few clouds, Partly cloudy, Partly cloudy"
    )
    temp: float = Field(
        description="Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)",
    )
    hum: float = Field(
        description="Normalized humidity. The values are divided to 100 (max)",
    )
    windspeed: float = Field(
        description="Normalized wind speed. The values are divided to 67 (max)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "season": 1,
                    "yr": 0,
                    "hr": 0,
                    "holiday": 0,
                    "weekday": 6,
                    "workingday": 0,
                    "weathersit": 1,
                    "temp": 0.24,
                    "hum": 0.81,
                    "windspeed": 0.0,
                }
            ]
        }
    }

class ReloadModelInput(BaseModel):
    reload_model_name: str = Field(..., description="The name of the model to load from MLflow.")
    alias: str = Field(..., description="The alias of the model version to load.")

    class Config:
        schema_extra = {
            "example": {
                "reload_model_name": "bike_sharing_model_prod",
                "alias": "champion"
            }
        }

class ModelOutput(BaseModel):
    int_output: int = Field(
        description="The predicted count of total rental bikes",
    )
    model_name: str = Field(
        description="The name of the model used for prediction",
    )
    model_version: int = Field(
        description="The version of the model used for prediction",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 16,
                    "model_name": "bike_sharing_model_prod",
                    "model_version": 1
                }
            ]
        }
    }

model, version_model, data_dictionary = load_model("bike_sharing_model_prod", "champion")

app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def one_hot_encode(df: pd.DataFrame, one_hot_cols: list) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to specified columns of a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - one_hot_cols (list): List of column names to apply One-Hot Encoding to.
    
    Returns:
    - pd.DataFrame: The DataFrame with One-Hot Encoded columns.
    """
    # Apply One-Hot Encoding
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
    return df

def cyclic_encode(df: pd.DataFrame, columns: list, max_value: int = 23) -> pd.DataFrame:
    """
    Applies cyclic encoding to specified columns of a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to apply Cyclic Encoding to.
    - max_value (int): The maximum value the cyclic variable can take (e.g., 23 for hours).
    
    Returns:
    - pd.DataFrame: The DataFrame with Cyclic Encoded columns added.
    """
    for col_name in columns:
        # Compute the sin and cos components
        df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
        df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
        # Drop the original column
        df = df.drop(col_name, axis=1)
        
    return df


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/reload_model")
def reload_model(
    reload_input: ReloadModelInput = Body(...),
):
    """
    Endpoint to reload the model with the given name and alias.
    
    :param model_name: The name of the model to load from MLflow.
    :param alias: The alias of the model version to load.
    """
    global model, model_name, version_model, data_dictionary

    try:
        model_name = reload_input.reload_model_name
        model, version_model, data_dictionary = load_model(model_name, reload_input.alias)
        print(f"Model {model_name} reloaded with version {version_model} from MLflow")
        return JSONResponse(content={"message": f"Model {model_name} reloaded with version {version_model}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.post("/predict", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks,
):
    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]
    
    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape(1, -1), columns=features_key)
    
    # Process categorical features
    for categorical_column in data_dictionary["one_hot_encoded_columns"] :
        features_df[categorical_column] = features_df[categorical_column].astype(int)
        categories = data_dictionary["categories_values_per_categorical"][categorical_column]
        features_df[categorical_column] = pd.Categorical(features_df[categorical_column], categories=categories)
    
    features_df = one_hot_encode(features_df, data_dictionary["one_hot_encoded_columns"])
    features_df = cyclic_encode(features_df, data_dictionary["cyclic_encoded_columns"])
    
    # Standardize the features
    features_df = (features_df - data_dictionary["standard_scaler_mean"]) / data_dictionary["standard_scaler_std"]
    
    # Make the prediction
    prediction = model.predict(features_df)
    
    # Compute the reverse transformation
    prediction = np.exp(prediction).astype(int)
    
    return ModelOutput(int_output=prediction, model_name=model_name, model_version=version_model)