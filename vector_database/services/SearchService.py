from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from vector_database.exceptions import CollectionDoesNotExistError
from vector_database.models import SearchItemRequest
from vector_database.models_responses import PayloadResponse, ScoredPointResponse
from vector_database.services.ItemService import ItemService
from datetime import datetime


class SearchService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def search(self, name: str, request: SearchItemRequest) -> list[ScoredPointResponse]:
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        qdrant_filter = SearchService.create_filter(request)
        response = self.client.query_points(collection_name=name, query=request.vector, query_filter=qdrant_filter,
            limit=request.top_k)

        return SearchService.format_points(response.points)

    @staticmethod
    def create_filter(request: SearchItemRequest) -> Filter:
        conditions = []

        author = request.filter.get("author")
        if author:
            conditions.append(FieldCondition(key="author", match=MatchValue(value=author)))

        start = request.filter.get("starting_date")
        end = request.filter.get("ending_date")
        if start or end:
            range_args = {}
            if start:
                range_args["gte"] = ItemService.convert_date_string_to_int(start)
            if end:
                range_args["lte"] = ItemService.convert_date_string_to_int(end)
            conditions.append(FieldCondition(key="published_at", range=Range(**range_args)))

        if conditions:
            qdrant_filter = Filter(must=conditions)
        else:
            qdrant_filter = None

        return qdrant_filter

    @staticmethod
    def format_points(points: list[ScoredPoint]) -> list[ScoredPointResponse]:
        formatted_points = []

        for point in points:
            point_info = point.model_dump()

            payload = point_info["payload"]
            date_obj = datetime.strptime(str(payload["published_at"]), "%Y%m%d")
            payload["published_at"] = date_obj.strftime("%Y-%m-%d")
            formatted_payload = PayloadResponse(**payload)

            formatted_points.append(ScoredPointResponse(score=point_info["score"], payload=formatted_payload))

        return formatted_points
