from fastapi import FastAPI
from backend.router import users


app = FastAPI()
app.include_router(users.router)
