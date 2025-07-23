import os

from pydantic import BaseModel, ConfigDict, Field

from entity_graph.models.entity_graph.entity import Entity
from entity_graph.models.entity_graph.graph import GraphNodeType


class IdentifierData(BaseModel):
    values: set = Field(default_factory=set, description="Values of the identifier")
    enrichment: dict = Field(
        default_factory=dict, description="Entities for enrichment"
    )


class Identifier(Entity):

    def __init__(self, entity_id, name, data):
        super().__init__(entity_id, name)
        self.entity_type = GraphNodeType.IDENTIFIER  # node type
        self.data = data

    def to_dict(self):
        data = super().to_dict()
        data_dump = {k: v if k != "values" else list(v) for k, v in self.data.items()}
        data.update({"data": data_dump})
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["entity_id"], data["name"], data["data"])
        obj.relations = data.get("relations", [])
        return obj
