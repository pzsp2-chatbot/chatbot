from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from vector_database.exceptions import CollectionDoesNotExistError
from vector_database.models import SearchItemRequest
from vector_database.models_responses import PayloadResponse, ScoredPointResponse
from vector_database.services.ItemService import ItemService


class SearchService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def search(
        self, name: str, request: SearchItemRequest
    ) -> list[ScoredPointResponse]:
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

        response = self.client.query_points(
            collection_name=name,
            query=request.vector,
            query_filter=qdrant_filter,
            limit=request.top_k,
        )

        return response.points

    @staticmethod
    def create_filter(request: SearchItemRequest) -> Filter | None:
        conditions = []

        author_cond = SearchService.create_author_condition(request)
        if author_cond:
            conditions.append(author_cond)

        date_cond = SearchService.create_date_condition(request)
        if date_cond:
            conditions.append(date_cond)

        return Filter(must=conditions) if conditions else None

    @staticmethod
    def create_author_condition(request: SearchItemRequest) -> FieldCondition | None:
        author = request.filter.get("author")
        if author:
            return FieldCondition(key="author", match=MatchValue(value=author))
        return None

    @staticmethod
    def create_date_condition(request: SearchItemRequest) -> FieldCondition | None:
        start = request.filter.get("starting_date")
        end = request.filter.get("ending_date")
        if not start and not end:
            return None

        range_args = {}
        if start:
            range_args["gte"] = ItemService.convert_date_string_to_int(start)
        if end:
            range_args["lte"] = ItemService.convert_date_string_to_int(end)

        return FieldCondition(key="published_at", range=Range(**range_args))

    @staticmethod
    def format_points(points: list[ScoredPoint]) -> list[ScoredPointResponse]:
        formatted_points = []

        for point in points:
            point_info = point.model_dump()

            payload = point_info["payload"]
            date_obj = datetime.strptime(str(payload["published_at"]), "%Y%m%d")
            payload["published_at"] = date_obj.strftime("%Y-%m-%d")
            formatted_payload = PayloadResponse(**payload)

            formatted_points.append(
                ScoredPointResponse(
                    score=point_info["score"], payload=formatted_payload
                )
            )

        return formatted_points
