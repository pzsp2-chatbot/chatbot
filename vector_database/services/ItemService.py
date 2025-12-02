from datetime import datetime
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from vector_database.exceptions import CollectionDoesNotExistError, DocumentDoesNotExistError, InvalidDateFormatError, \
    InputDataError
from vector_database.models import AddItemRequest, PayloadDict


class ItemService:
    def __init__(self, client: QdrantClient):
        self.client = client


    def add_item(self, name: str, request: AddItemRequest) -> str:
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        qdrant_id = str(uuid.uuid4())
        payload_with_id = ItemService.prepare_payload(request.payload)

        self.client.upsert(collection_name=name, points=[{"id": qdrant_id, "vector": request.vector,
            "payload": payload_with_id}])

        return f"Item added to collection '{name}'."


    @staticmethod
    def prepare_payload(payload: PayloadDict) -> dict:
        if any(not value for value in payload.values()):
            raise InputDataError("At least one mandatory field is empty.")

        if payload["authors"] == [] or payload["author_affiliations"] == []:
            raise InputDataError("At least one mandatory field is empty.")

        if len(payload["authors"]) != len(payload["author_affiliations"]):
            raise InputDataError("Number of authors and affiliations do not match.")

        document_id = str(uuid.uuid4())
        payload_with_id = dict(payload)
        payload_with_id["document_id"] = document_id

        created = ItemService.convert_date_string_to_int(payload_with_id["created"])
        modified = ItemService.convert_date_string_to_int(payload_with_id["modified"])

        if created > modified:
            raise InputDataError("Created date is newer than modified date.")

        payload_with_id["created"] = created
        payload_with_id["modified"] = modified

        return payload_with_id


    @staticmethod
    def convert_date_string_to_int(date: str) -> int:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise InvalidDateFormatError("Date must be in YYYY-MM-DD format.")

        return parsed_date.year * 10000 + parsed_date.month * 100 + parsed_date.day


    def delete_item(self, name: str, document_id: str) -> str:
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        points, _ = self.client.scroll(collection_name=name, scroll_filter=Filter(must=[FieldCondition(
                        key="document_id", match=MatchValue(value=document_id))]), limit=1)
        if len(points) == 0:
            raise DocumentDoesNotExistError(f"Item with document id '{document_id}' not found.")

        self.client.delete(collection_name=name, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])))

        return f"Item with document_id {document_id} deleted from collection {name}."
