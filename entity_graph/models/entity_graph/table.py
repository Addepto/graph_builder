import os

from entity_graph.models.entity_graph.entity import Entity
from entity_graph.models.entity_graph.graph import GraphNodeType


class Table(Entity):

    def __init__(self, entity_id, name, columns, identifiers, page):
        super().__init__(entity_id, name)
        self.entity_type = GraphNodeType.TABLE  # node type
        self.columns = columns  # List of column names
        self.identifiers = identifiers  # dict mapping columns to identifiers
        self.page = page
        self.header = {}
        self.header_identifiers = {}  # dict mapping columns to identifiers

    def to_dict(self):
        data = super().to_dict()
        data.update({"columns": self.columns})
        data.update({"identifiers": self.identifiers})
        data.update({"page": self.page})
        data.update({"header": self.header})
        data.update({"header_identifiers": self.header_identifiers})
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data["entity_id"],
            data["name"],
            data["columns"],
            data["identifiers"],
            data["page"],
        )
        obj.relations = data.get("relations", [])
        obj.header = data.get("header", {})
        obj.header_identifiers = data.get("header_identifiers", {})
        return obj

    def get_rows_ids(self):
        return self.get_relations("in_table")
