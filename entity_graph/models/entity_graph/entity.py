import json
import os
import uuid

from entity_graph.models.entity_graph.graph import GraphNodeType


class Entity:

    def __init__(self, entity_id, name):
        if entity_id is None:
            entity_id = str(uuid.uuid4())
        self.entity_id = entity_id
        self.name = name
        self.relations = []  # List of tuples (relation_type, entity_id, relation_data)
        self.entity_type: GraphNodeType = GraphNodeType.ENTITY  # node type

    def _get_relations_keys(self):
        return {(rtype, rid): rdata for rtype, rid, rdata in self.relations}

    def get_relations_types(self):
        return [rtype for rtype, _, _ in self.relations]

    def add_relation(self, other_entity, relation_type, r_data=None):
        """Add relation to other_entity (one way only!)"""
        if isinstance(other_entity, str):
            other_entity_id = other_entity
        else:
            other_entity_id = other_entity.entity_id

        if (relation_type, other_entity_id) not in self._get_relations_keys():
            self.relations.append((relation_type, other_entity_id, r_data))
        else:
            # TODO logger?
            print(
                f"WARNING: duplicated relation for {self.name} {relation_type}. Ignore  re new data?"
            )  # TODO:

    def get_relations(self, relation_type):
        return [eid for rel_type, eid, _ in self.relations if rel_type == relation_type]

    def to_dict(self):
        """Convert entity to dictionary format."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "relations": self.relations,
            "entity_type": self.entity_type,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["entity_id"], data["name"])
        obj.relations = data.get("relations", [])
        return obj
