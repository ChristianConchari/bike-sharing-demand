import os
from utils import load_model

# Define the initial state variables
model_name = "bike_sharing_model_prod"
version_model = 0
data_dictionary = {}

# Load the initial model and update state variables
model, version_model, data_dictionary = load_model(model_name, "champion")