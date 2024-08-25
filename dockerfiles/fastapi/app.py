import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
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
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0
        
    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

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

    datetime: str = Field(
        description="The hourly date and the tmestamp",
        format="date-time",
    )
    season: Literal[1, 2, 3, 4] = Field(
        description="The season of the year. 1: winter; 2: spring; 3: summer; 4: fall"
    )
    holiday: Literal[0, 1] = Field(
        description="Whether the day is a holiday or not. 1: holiday; 0: no holiday"
    )
    workingday: Literal[0, 1] = Field(
        description="Whether the day is a working day or not. 1: working day; 0: no working day"
    )
    weather: Literal[1, 2, 3, 4] = Field(
        description=(
            "Weather situation:\n"
            "1: Clear, Few clouds, Partly cloudy, Partly cloudy\n"
            "2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist\n"
            "3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds\n"
            "4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"
        )
    )
    temp: float = Field(
        description="Temperature in Celsius",
        ge=0.0,
        le=50.0,
    )
    humidity: float = Field(
        description="Relative humidity",
        ge=0.0,
        le=100.0,
    )
    windspeed: float = Field(
        description="Wind speed",
        ge=0.0,
    )
    casual: int = Field(
        description="Number of casual users",
        ge=0,
    )
    registered: int = Field(
        description="Number of registered users",
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "datetime": "2011-01-01 00:00:00",
                    "season": 1,
                    "holiday": 0,
                    "workingday": 0,
                    "weather": 1,
                    "temp": 9.84,
                    "humidity": 81.0,
                    "windspeed": 0.0,
                    "casual": 3,
                    "registered": 13
                }
            ]
        }
    }

class ModelOutput(BaseModel):
    int_output: int = Field(
        description="The predicted count of total rental bikes",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 16
                }
            ]
        }
    }

model, version_model, data_dictionary = load_model("bike_sharing_model_prod ", "best-model")

app = FastAPI()


@app.get("/")
async def read_root():
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Heart Disease Detector API"}))

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
    