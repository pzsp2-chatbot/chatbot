import os
import warnings

import pydantic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from vector_database.exceptions import (
    CollectionAlreadyExistsError,
    CollectionDoesNotExistError,
    DocumentDoesNotExistError,
    InvalidDateFormatError,
)
from vector_database.models import (
    AddItemRequest,
    CreateCollectionRequest,
    SearchItemRequest,
)
from vector_database.services.CollectionService import CollectionService
from vector_database.services.ItemService import ItemService
from vector_database.services.SearchService import SearchService

warnings.filterwarnings(
    "ignore", message="Api key is used with an insecure connection."
)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
app = FastAPI()
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", api_key=QDRANT_API_KEY
)
collection_service = CollectionService(qdrant_client)
item_service = ItemService(qdrant_client)
search_service = SearchService(qdrant_client)


@app.get("/")
def read_root():
    return {"message": "This is a RAG app"}


@app.get("/collections")
def get_collections():
    try:
        collections = collection_service.get_collections()
        return {"status": "ok", "collections": collections}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )


@app.post("/collections")
def create_collection(request: CreateCollectionRequest):
    try:
        message = collection_service.create_collection(request)
        return {"status": "ok", "message": message}
    except CollectionAlreadyExistsError as e:
        raise HTTPException(
            status_code=400, detail={"status": "bad request", "message": str(e)}
        )
    except pydantic.ValidationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )


@app.post("/collections/{collection_name}/items")
def add_item(collection_name: str, request: AddItemRequest):
    try:
        message = item_service.add_item(collection_name, request)
        return {"status": "ok", "message": message}
    except CollectionAlreadyExistsError as e:
        raise HTTPException(
            status_code=400, detail={"status": "bad request", "message": str(e)}
        )
    except InvalidDateFormatError as e:
        raise HTTPException(
            status_code=422,
            detail={"status": "invalid input data format", "message": str(e)},
        )
    except pydantic.ValidationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )


@app.post("/collections/{collection_name}/search")
def search(collection_name: str, request: SearchItemRequest):
    try:
        items = search_service.search(collection_name, request)
        return {"status": "ok", "items": items}
    except CollectionDoesNotExistError as e:
        raise HTTPException(
            status_code=404, detail={"status": "not found", "message": str(e)}
        )
    except InvalidDateFormatError as e:
        raise HTTPException(
            status_code=422,
            detail={"status": "invalid input data format", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )


@app.delete("/collections/{collection_name}/items/{document_id}")
def delete_item(collection_name: str, document_id: str):
    try:
        message = item_service.delete_item(collection_name, document_id)
        return {"status": "ok", "message": message}
    except CollectionDoesNotExistError as e:
        raise HTTPException(
            status_code=404, detail={"status": "not found", "message": str(e)}
        )
    except DocumentDoesNotExistError as e:
        raise HTTPException(
            status_code=404, detail={"status": "not found", "message": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    try:
        message = collection_service.delete_collection(collection_name)
        return {"status": "ok", "message": message}
    except CollectionDoesNotExistError as e:
        raise HTTPException(
            status_code=404, detail={"status": "not found", "message": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": str(e)}
        )
