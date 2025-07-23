import os

import pandas as pd

from entity_graph.models.entity_graph.entity import Entity
from entity_graph.models.entity_graph.graph import GraphNodeType


def serialize(v):
    if isinstance(v, float) and pd.isna(v):
        return None

    return v


class Object(Entity):
    """Object is assumed to be extracted from an artifact using an identifier"""

    def __init__(
        self, entity_id, name, identifier_type, attributes, extraction_metadata
    ):
        super().__init__(entity_id, name)
        self.entity_type = GraphNodeType.OBJECT  # node type
        self.attributes = attributes  # Dictionary of attributes
        self.identifier_type = identifier_type
        self.extraction_metadata = extraction_metadata

    def to_dict(self):
        data = super().to_dict()
        self.attributes = {k: serialize(v) for k, v in self.attributes.items()}
        data.update(
            {
                "attributes": self.attributes,
                "identifier_type": self.identifier_type,
                "extraction_metadata": self.extraction_metadata,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data["entity_id"],
            data["name"],
            data["identifier_type"],
            data["attributes"],
            data["extraction_metadata"],
        )
        obj.relations = data.get("relations", [])
        return obj
