from abc import ABC, abstractmethod
from typing import List
from embeddings.models.article import Article

class IArticleLoader(ABC):
    @abstractmethod
    def load_all(self) -> List[Article]:
        pass
