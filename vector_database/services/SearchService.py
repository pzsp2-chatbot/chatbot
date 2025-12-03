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

        return SearchService.format_points(response.points)

    @staticmethod
    def format_points(points: list[ScoredPoint]) -> list[ScoredPointResponse]:
        formatted_points = []

        for point in points:
            point_info = point.model_dump()

            payload = point_info["payload"]
            date_obj = datetime.strptime(str(payload["created"]), "%Y%m%d")
            payload["created"] = date_obj.strftime("%Y-%m-%d")
            date_obj = datetime.strptime(str(payload["modified"]), "%Y%m%d")
            payload["modified"] = date_obj.strftime("%Y-%m-%d")
            formatted_payload = PayloadResponse(**payload)

            formatted_points.append(ScoredPointResponse(score=point_info["score"], payload=formatted_payload))

        return formatted_points


class FilterCreator:
    @staticmethod
    def create_filter(request: SearchItemRequest) -> Filter | None:
        conditions = []

        title_cond = FilterCreator.create_simple_condition(request, "title")
        if title_cond:
            conditions.append(title_cond)

        date_create_cond = FilterCreator.create_date_condition(request, "creation")
        if date_create_cond:
            conditions.append(date_create_cond)

        date_modify_cond = FilterCreator.create_date_condition(request, "modification")
        if date_modify_cond:
            conditions.append(date_modify_cond)

        language_cond = FilterCreator.create_simple_condition(request, "language")
        if language_cond:
            conditions.append(language_cond)

        doi_cond = FilterCreator.create_simple_condition(request, "doi")
        if doi_cond:
            conditions.append(doi_cond)

        url_cond = FilterCreator.create_simple_condition(request, "url")
        if url_cond:
            conditions.append(url_cond)

        authors_cond = FilterCreator.create_list_condition(request, "authors")
        if authors_cond:
            conditions.append(authors_cond)

        author_affiliations_cond = FilterCreator.create_list_condition(request, "author_affiliations")
        if author_affiliations_cond:
            conditions.append(author_affiliations_cond)

        return Filter(must=conditions) if conditions else None

    @staticmethod
    def create_simple_condition(request: SearchItemRequest, key: str) -> FieldCondition | None:
        value = request.filter.get(key)
        if value:
            return FieldCondition(key=key, match=MatchValue(value=value))
        return None

    @staticmethod
    def create_date_condition(request: SearchItemRequest, field_name: str) -> FieldCondition | None:
        start = request.filter.get(f"starting_{field_name}_date")
        end = request.filter.get(f"ending_{field_name}_date")
        if not start and not end:
            return None

        range_args = {}
        if start:
            range_args["gte"] = ItemService.convert_date_string_to_int(start)
        if end:
            range_args["lte"] = ItemService.convert_date_string_to_int(end)

        name = "created" if field_name == "creation" else "modified"
        return FieldCondition(key=name, range=Range(**range_args))

    @staticmethod
    def create_list_condition(request: SearchItemRequest, key: str) -> Filter | None:
        collection = request.filter.get(key)

        if not collection:
            return None

        conditions = [FieldCondition(key=key, match=MatchValue(value=element))
                      for element in collection]

        return Filter(must=conditions)
