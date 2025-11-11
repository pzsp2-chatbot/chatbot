import os
import pydantic
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from vector_database.exceptions import *
from vector_database.models import CreateCollectionRequest, AddItemRequest, SearchItemRequest
from dotenv import load_dotenv
from vector_database.services.CollectionService import CollectionService
from vector_database.services.ItemService import ItemService
from vector_database.services.SearchService import SearchService

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
app = FastAPI()
qdrant_client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", api_key=QDRANT_API_KEY)
collection_service = CollectionService(qdrant_client)
item_service = ItemService(qdrant_client)
search_service = SearchService(qdrant_client)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/collections")
def create_collection(request: CreateCollectionRequest):
    try:
        message = collection_service.create_collection(request.name, request.vector_size)
        return {"status": "ok", "message": message}
    except CollectionAlreadyExistsError as e:
        raise HTTPException(status_code=400, detail={"status": "bad request", "message": str(e)})
    except pydantic.ValidationError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post("/collections/{collection_name}/items")
def add_item(collection_name: str, request: AddItemRequest):
    try:
        message = item_service.add_item(collection_name, request)
        return {"status": "ok", "message": message}
    except CollectionAlreadyExistsError as e:
        raise HTTPException(status_code=400, detail={"status": "bad request", "message": str(e)})
    except pydantic.ValidationError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


@app.post("/collections/{collection_name}/search")
def search(collection_name: str, request: SearchItemRequest):
    pass


@app.delete("/collections/{collection_name}/items/{item_id}")
def delete_item(collection_name: str, item_id: int):
    pass


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    try:
        message = collection_service.delete_collection(collection_name)
        return {"status": "ok", "message": message}
    except CollectionDoesNotExistError as e:
        raise HTTPException(status_code=404, detail={"status": "not found", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
