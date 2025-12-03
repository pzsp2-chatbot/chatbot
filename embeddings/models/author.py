from dataclasses import dataclass
from typing import Optional


@dataclass
class Author:
    full_name: str
    affiliation: Optional[str]
