from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict
import requests

from pydantic import BaseModel

# source ./venv/Scripts/activate
# uvicorn server:app --reload

# http://127.0.0.1:8000/docs

app = FastAPI()

class Info(BaseModel):
    tunnel_name: str
    accident_type: int
    latitude: float
    longitude: float

@app.post("/accident/")
async def accident_post(data: Info):
    return data
    
