from fastapi import FastAPI

from vector_database.models import CreateCollectionRequest

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/create_collection")
def create_collection(request: CreateCollectionRequest):
    pass
