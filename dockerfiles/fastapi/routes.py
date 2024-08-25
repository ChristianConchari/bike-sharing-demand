from fastapi import APIRouter, Body, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from models import ModelInput, ReloadModelInput, ModelOutput
from utils import load_model, one_hot_encode, cyclic_encode
import config
import logging
import pandas as pd
import numpy as np

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/reload_model")
def reload_model(reload_input: ReloadModelInput = Body(...)) -> JSONResponse:
    """
    Endpoint to reload the model with the given name and alias from MLflow.

    Parameters:
    - reload_input (ReloadModelInput): Pydantic model containing the model name and alias to reload.

    Returns:
    - JSONResponse: A response containing a success message with the reloaded model name and version.

    Raises:
    - HTTPException: If the model fails to reload.
    """
    # Initialize logging
    logger = logging.getLogger(__name__)
    logger.info("Received request to reload model")
    
    try:
        # Attempt to reload the model with the specified name and alias
        logger.info(f"Attempting to reload model '{reload_input.reload_model_name}' with alias '{reload_input.alias}'")
        
        # Update model state in config
        config.model_name = reload_input.reload_model_name
        config.model, config.version_model, config.data_dictionary = load_model(config.model_name, reload_input.alias)
        
        logger.info(f"Model '{config.model_name}' reloaded successfully with version {config.version_model} from MLflow")
        return JSONResponse(content={"message": f"Model {config.model_name} reloaded with version {config.version_model} from MLflow"})
    
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflowException occurred while reloading the model: {e}")
        raise HTTPException(status_code=500, detail=f"MLflowException: Failed to reload model: {str(e)}")
    
    except FileNotFoundError as e:
        logger.error(f"File not found while loading model or data: {e}")
        raise HTTPException(status_code=404, detail=f"FileNotFoundError: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error occurred while reloading the model: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/predict", response_model=ModelOutput)
def predict(
    background_tasks: BackgroundTasks, 
    features: ModelInput = Body(embed=True)
) -> ModelOutput:
    """
    Endpoint to predict bike sharing demand based on input features.

    Parameters:
    - background_tasks (BackgroundTasks): FastAPI BackgroundTasks instance for any background processing (if needed).
    - features (ModelInput): Pydantic model containing the input features for prediction.

    Returns:
    - ModelOutput: Pydantic model containing the prediction result, model name, and model version.

    Raises:
    - HTTPException: If an error occurs during data processing or model prediction.
    """
    # Initialize logging
    logger = logging.getLogger(__name__)
    logger.info("Received request for prediction")

    try:
        # Access the model and data dictionary from the config
        model = config.model
        data_dictionary = config.data_dictionary

        logger.info("Extracting features from the request")
        features_list = [*features.dict().values()]
        features_key = [*features.dict().keys()]

        logger.info("Converting features into a pandas DataFrame")
        features_df = pd.DataFrame(np.array(features_list).reshape(1, -1), columns=features_key)

        logger.info("Processing categorical features for one-hot encoding and cyclic encoding")
        # Process categorical features for one-hot encoding
        for categorical_column in data_dictionary["one_hot_encoded_columns"]:
            features_df[categorical_column] = features_df[categorical_column].astype(int)
            categories = data_dictionary["categories_values_per_categorical"][categorical_column]
            features_df[categorical_column] = pd.Categorical(features_df[categorical_column], categories=categories)

        features_df = one_hot_encode(features_df, data_dictionary["one_hot_encoded_columns"])
        features_df = cyclic_encode(features_df, data_dictionary["cyclic_encoded_columns"])

        logger.info("Standardizing features using the provided mean and standard deviation")
        # Standardize the features
        features_df = (features_df - data_dictionary["standard_scaler_mean"]) / data_dictionary["standard_scaler_std"]

        logger.info("Making prediction using the loaded model")
        # Make the prediction
        prediction = model.predict(features_df)
        prediction = np.exp(prediction).astype(int)

        logger.info(f"Prediction made successfully: {prediction}")

        return ModelOutput(int_output=prediction, model_name=config.model_name, model_version=config.version_model)

    except KeyError as e:
        logger.error(f"KeyError: Missing or unexpected data: {e}")
        raise HTTPException(status_code=400, detail=f"KeyError: Missing or unexpected data: {str(e)}")

    except ValueError as e:
        logger.error(f"ValueError: Error in data conversion or prediction: {e}")
        raise HTTPException(status_code=422, detail=f"ValueError: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

