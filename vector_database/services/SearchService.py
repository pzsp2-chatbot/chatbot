from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from vector_database.exceptions import CollectionDoesNotExistError
from vector_database.models import SearchItemRequest
from vector_database.services.ItemService import ItemService


class SearchService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def search(self, name: str, request: SearchItemRequest) -> list[ScoredPoint]:
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        qdrant_filter = SearchService.create_filter(request)

        response = self.client.query_points(
            collection_name=name,
            query=request.vector,
            query_filter=qdrant_filter,
            limit=request.top_k,
        )

        return response.points

    @staticmethod
    def create_filter(request: SearchItemRequest) -> Filter:
        conditions = []

        author = request.filter.get("author")
        if author:
            conditions.append(
                FieldCondition(key="author", match=MatchValue(value=author))
            )

        start = request.filter.get("starting_date")
        end = request.filter.get("ending_date")
        if start or end:
            range_args = {}
            if start:
                range_args["gte"] = ItemService.convert_date_string_to_int(start)
            if end:
                range_args["lte"] = ItemService.convert_date_string_to_int(end)
            conditions.append(
                FieldCondition(key="published_at", range=Range(**range_args))
            )

        if conditions:
            qdrant_filter = Filter(must=conditions)
        else:
            qdrant_filter = None

        return qdrant_filter
