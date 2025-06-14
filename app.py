from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import random

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/change-message")
async def change_message():
    messages = [
        "Hello World!",
        "Upside Down",
        "Around the World",
        "Right Side Up",
        "Left Side Up"
    ]
    return {"message": random.choice(messages)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 