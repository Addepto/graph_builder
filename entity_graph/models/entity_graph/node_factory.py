from entity_graph.database.schemas.graph_node.models import GraphNodeModel
from entity_graph.models.entity_graph.artifact import Artifact
from entity_graph.models.entity_graph.entity import Entity
from entity_graph.models.entity_graph.file import File
from entity_graph.models.entity_graph.graph import GraphNodeType
from entity_graph.models.entity_graph.identifier import Identifier
from entity_graph.models.entity_graph.object import Object
from entity_graph.models.entity_graph.table import Table

CLASS_MAP = {
    GraphNodeType.ENTITY: Entity,
    GraphNodeType.ARTIFACT: Artifact,
    GraphNodeType.FILE: File,
    GraphNodeType.OBJECT: Object,
    GraphNodeType.TABLE: Table,
    GraphNodeType.IDENTIFIER: Identifier,
}


def node_factory(graph_node: GraphNodeModel):
    if graph_node.node_type in CLASS_MAP:
        return CLASS_MAP[graph_node.node_type].from_dict(graph_node.node_data)
    raise ValueError(f"Node type {graph_node.node_type} is not supported.")


def node_from_dict(graph_node_data: dict):
    ntype = graph_node_data.get("entity_type")
    if ntype in CLASS_MAP:
        return CLASS_MAP[ntype].from_dict(graph_node_data)
    raise ValueError(f"Node type {ntype} is not supported.")
