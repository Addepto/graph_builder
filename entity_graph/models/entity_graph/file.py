import os

from entity_graph.models.entity_graph.entity import Entity
from entity_graph.models.entity_graph.graph import GraphNodeType


class File(Entity):

    def __init__(self, entity_id, name, file_type, extension, data_type, size):
        super().__init__(entity_id, name)
        self.entity_type = GraphNodeType.FILE  # node type
        self.file_type = file_type
        self.extension = extension
        self.data_type = data_type
        self.size = size  # File size in bytes

    def to_dict(self):
        data = super().to_dict()
        data.update(
            {
                "file_type": self.file_type,
                "extension": self.extension,
                "data_type": self.data_type,
                "size": self.size,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data["entity_id"],
            data["name"],
            data["file_type"],
            data["extension"],
            data["data_type"],
            data["size"],
        )
        obj.relations = data.get("relations", [])
        return obj
