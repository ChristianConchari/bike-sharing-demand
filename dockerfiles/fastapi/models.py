from typing import Literal
from pydantic import BaseModel, Field

class ModelInput(BaseModel):
    season: Literal[1, 2, 3, 4] = Field(description="The season of the year. 1: winter; 2: spring; 3: summer; 4: fall")
    yr: int = Field(description="The year (0: 2011, 1: 2012)", ge=0, le=1)
    hr: int = Field(description="The hour of the day (0 to 23)", ge=0, le=23)
    holiday: Literal[0, 1] = Field(description="Whether the day is a holiday or not. 1: holiday; 0: no holiday")
    weekday: int = Field(description="The day of the week", ge=0, le=6)
    workingday: Literal[0, 1] = Field(description="Whether the day is a working day or not. 1: working day; 0: no working day")
    weathersit: Literal[1, 2, 3, 4] = Field(description="Clear, Few clouds, Partly cloudy, Partly cloudy")
    temp: float = Field(description="Normalized temperature in Celsius.")
    hum: float = Field(description="Normalized humidity.")
    windspeed: float = Field(description="Normalized wind speed.")

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
            }]
        }
    }


class ReloadModelInput(BaseModel):
    reload_model_name: str = Field(..., description="The name of the model to load from MLflow.")
    alias: str = Field(..., description="The alias of the model version to load.")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "reload_model_name": "bike_sharing_model_prod",
                "alias": "champion"
            }]
        }
    }


class ModelOutput(BaseModel):
    int_output: int = Field(description="The predicted count of total rental bikes")
    model_name: str = Field(description="The name of the model used for prediction")
    model_version: int = Field(description="The version of the model used for prediction")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "int_output": 16,
                "model_name": "bike_sharing_model_prod",
                "model_version": 1
            }]
        }
    }
