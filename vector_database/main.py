from fastapi import FastAPI

from vector_database.models import CreateCollectionRequest, AddItemRequest, SearchItemRequest

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/collections")
def create_collection(request: CreateCollectionRequest):
    pass


@app.post("/collections/{collection_name}/items")
def add_item(collection_name: str, request: AddItemRequest):
    pass


@app.post("/collections/{collection_name}/search")
def search(collection_name: str, request: SearchItemRequest):
    pass
