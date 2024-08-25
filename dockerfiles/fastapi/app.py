from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from config import model, version_model, data_dictionary
from routes import router
from os import path, makedirs

app = FastAPI()

# Setup static files and templates
if not path.exists("static"):
    makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Include routes from the router
app.include_router(router)
