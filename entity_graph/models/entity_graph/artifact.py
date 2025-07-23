import pandas as pd
from pydantic import BaseModel, Field

from entity_graph.models.entity_graph.entity import Entity
from entity_graph.models.entity_graph.graph import GraphNodeType


class ArtifactData(BaseModel): ...


class RowArtifactData(ArtifactData, extra="allow"): ...


class IdentifierArtifactData(ArtifactData):
    values: set = Field(default_factory=set, description="Values of the identifier")


def serialize(v):
    if isinstance(v, float) and pd.isna(v):
        return None

    return v


class Artifact(Entity):

    def __init__(self, entity_id, name, data, normalized_data, data_type):
        super().__init__(entity_id, name)
        self.entity_type = GraphNodeType.ARTIFACT  # node type
        # TODO: consider one data dict with normalization keys
        self.data = data
        self.normalized_data = normalized_data
        self.data_type = data_type  # inferited from file, e.g. instance/model

    def to_dict(self):
        data = super().to_dict()
        self.data = {k: serialize(v) for k, v in self.data.items()}
        data.update(
            {
                "data": self.data,
                "normalized_data": self.normalized_data,
                "data_type": self.data_type,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data["entity_id"],
            data["name"],
            data["data"],
            data["normalized_data"],
            data["data_type"],
        )
        obj.relations = data.get("relations", [])
        return obj
