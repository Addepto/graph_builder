from abc import ABC
from enum import Enum, StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GraphNodeType(StrEnum):
    ENTITY = "entity"
    FILE = "file"
    TABLE = "table"
    ARTIFACT = "artifact"
    OBJECT = "object"
    IDENTIFIER = "identifier"


class GraphResponse(BaseModel):
    nodes: dict[str, dict]
    edges: dict[tuple[str, str], dict]
