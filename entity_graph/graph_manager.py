from entity_graph.config import setup_logger
from entity_graph.models.entity_graph import Object

logger = setup_logger(__name__)


class EntityManager:

    def __init__(self, default_collection: str = "default"):
        self.entities = {}
        self.name_map = {}
        self.collection_name_maps = {}
        self.default_collection = default_collection

    def refresh_name_map(self, collection: str = None):
        """
        Refresh the name to entity mapping, optionally filtering by collection

        :param collection: The collection to filter by (None for all collections)
        :return: Updated name map
        """
        if collection is None:
            self.name_map = {}
            for obj in self.entities.values():
                if not hasattr(obj, "name"):
                    continue
                self.name_map[obj.name] = obj
            logger.debug(
                f"Creating name map, from {len(self.entities)} objects created {len(self.name_map)} unique keys"
            )
            return self.name_map

        filtered_map = {}
        for obj in self.entities.values():
            if not hasattr(obj, "name"):
                continue

            obj_collection = getattr(obj, "collection", self.default_collection)
            if obj_collection == collection:
                filtered_map[obj.name] = obj

        self.collection_name_maps[collection] = filtered_map
        logger.debug(
            f"Creating collection name map, from {len(self.entities)} objects created {len(filtered_map)} unique keys"
        )
        return filtered_map

    def get_name_map(self, collection: str = None):
        """
        Get the name map for a specific collection

        :param collection: Collection name (None for full name map)
        :return: Dictionary mapping names to entities
        """
        if collection is None:
            if not self.name_map:
                self.refresh_name_map()
            return self.name_map

        if collection not in self.collection_name_maps:
            self.refresh_name_map(collection)

        return self.collection_name_maps.get(collection, {})

    def named_entity(self, name, collection: str = None):
        """
        Get an entity by name, optionally filtering by collection

        :param name: Name of the entity to retrieve
        :param collection: The collection to search in (None for all collections)
        :return: Entity with the specified name, or None if not found
        """
        name_map = self.get_name_map(collection)
        return name_map.get(name)

    def get_named_entity_relations(self, name, relation_type, collection: str = None):
        """
        Get relations of a named entity, optionally filtering by collection

        :param name: Name of the entity
        :param relation_type: Type of relation to filter by
        :param collection: The collection to search in
        :return: List of related entities
        """
        entity = self.find_entity(name, collection=collection)
        if not entity:
            return []

        related_ids = entity.get_relations(relation_type)

        if collection is None:
            return [self.get_entity(eid) for eid in related_ids]

        return [
            self.get_entity(eid)
            for eid in related_ids
            if getattr(self.get_entity(eid), "collection", self.default_collection)
            == collection
        ]

    def add_entity(self, entity, collection: str = None):
        """
        Add an entity to the manager

        :param entity: The entity to add
        :param collection: Collection to assign to the entity
        """
        if collection is not None:
            setattr(entity, "collection", collection)

        self.entities[entity.entity_id] = entity

        # Clear all name maps to force refresh
        self.name_map = {}
        self.collection_name_maps = {}

    def list_entities_by_collection(self, collection: str = None):
        """
        List entities filtered by collection

        :param collection: Collection to filter by (None for all)
        :return: List of entities
        """
        if collection is None:
            return list(self.entities.values())

        return [
            entity
            for entity in self.entities.values()
            if getattr(entity, "collection", self.default_collection) == collection
        ]

    def get_entity(self, entity_id):
        """Retrieve an entity by its ID."""
        return self.entities.get(entity_id)

    def find_entity(self, entity_name, filtering_func=None, collection: str = None):
        """
        Retrieve an entity by its name, optionally filtering by collection

        :param entity_name: Name of the entity to find
        :param filtering_func: Additional filtering function
        :param collection: The collection to search in
        :return: Entity with the specified name, or None if not found
        """
        if collection is not None:
            entity = self.named_entity(entity_name, collection)
            if entity and (filtering_func is None or filtering_func(entity)):
                return entity
            return None

        entities = []
        for entity in self.entities.values():
            if not hasattr(entity, "name") or entity.name != entity_name:
                continue

            if (
                collection is not None
                and getattr(entity, "collection", self.default_collection) != collection
            ):
                continue

            if filtering_func is not None and not filtering_func(entity):
                continue

            entities.append(entity)

        if len(entities) == 1:
            return entities[0]
        if len(entities) == 0:
            return None

        raise ValueError(
            f"Multiple entities with name {entity_name} found"
            + (f" in collection {collection}" if collection else "")
        )

    def add_relation(
        self, entity_id1, entity_id2, relation_type="edge", is_bidirectional=True
    ):
        """Create a relation between two entities."""
        entity1 = self.get_entity(entity_id1)
        entity2 = self.get_entity(entity_id2)
        if entity1 and entity2:
            if is_bidirectional:
                entity1.add_relation(entity2, relation_type)
                entity2.add_relation(
                    entity1, relation_type
                )  # Assuming bidirectional relationships
            else:
                raise ValueError("Unidirectional relationships not implemented")

    def get_related_entities(
        self, entity_id, relation_type=None, collection: str = None
    ):
        """
        Retrieve all related entities of a given entity

        :param entity_id: ID of the entity
        :param relation_type: Type of relation to filter by (None for all types)
        :param collection: The collection to filter related entities by
        :return: List of related entities
        """
        entity = self.get_entity(entity_id)
        if not entity:
            return []

        if relation_type:
            relations = [
                (rtype, eid, rdata)
                for rtype, eid, rdata in entity.relations
                if rtype == relation_type
            ]
        else:
            relations = entity.relations

        related_entities = []
        for _, eid, _ in relations:
            related_entity = self.get_entity(eid)
            if related_entity is None:
                continue

            if (
                collection is None
                or getattr(related_entity, "collection", self.default_collection)
                == collection
            ):
                related_entities.append(related_entity)

        return related_entities

    def get_table_rows(self, table, collection: str = None):
        """
        Get rows from a table, optionally filtering by collection

        :param table: The table to get rows from
        :param collection: The collection to filter rows by
        :return: List of row entities
        """
        row_ids = table.get_rows_ids()
        rows = []

        for row_id in row_ids:
            row = self.get_entity(row_id)
            if row is None:
                continue

            if (
                collection is None
                or getattr(row, "collection", self.default_collection) == collection
            ):
                rows.append(row)

        return rows

    def get_table_contents_dict(self, table, idx_col, collection: str = None):
        """
        Get table contents as a dictionary, optionally filtering by collection

        :param table: The table to get contents from
        :param idx_col: Column to use as index
        :param collection: The collection to filter contents by
        :return: Dictionary of table contents
        """
        row_ids = table.get_rows_ids()
        data = {}

        for row_id in row_ids:
            row = self.get_entity(row_id)
            if row is None or not hasattr(row, "data") or idx_col not in row.data:
                continue

            if (
                collection is not None
                and getattr(row, "collection", self.default_collection) != collection
            ):
                continue

            data[str(row.data[idx_col])] = {"source_entity": row_id, **row.data}

        return data

    def export_objects_graph(self, base_relation="parent_of", collection: str = None):
        """
        Export objects graph, optionally filtering by collection

        :param base_relation: Relation type to include
        :param collection: The collection to filter by
        :return: Tuple of nodes, edges, and metadata
        """
        objects = []
        for obj_id, obj in self.entities.items():
            if not (
                isinstance(obj, Object) or getattr(obj, "entity_type", None) == "object"
            ):
                continue

            if (
                collection is not None
                and getattr(obj, "collection", self.default_collection) != collection
            ):
                continue

            objects.append((obj_id, obj))

        nodes = [obj[1].name for obj in objects]

        metadata = {}
        for _, obj in objects:
            metadata[obj.name] = {
                **obj.attributes,
                **obj.extraction_metadata,
                "collection": getattr(obj, "collection", self.default_collection),
            }

        edges = []
        for _, obj in objects:
            parent_relations = obj.get_relations(base_relation)
            for o_id in parent_relations:
                related_entity = self.get_entity(o_id)
                if related_entity is None or related_entity.name not in nodes:
                    continue

                if (
                    collection is not None
                    and getattr(related_entity, "collection", self.default_collection)
                    != collection
                ):
                    continue

                edges.append((obj.name, related_entity.name))

        return nodes, edges, metadata

    def list_collections(self):
        """
        Get a list of all collections in use by the entities

        :return: List of collection names
        """
        collections = set()
        for entity in self.entities.values():
            coll = getattr(entity, "collection", self.default_collection)
            if coll:
                collections.add(coll)

        return sorted(list(collections))

    def __repr__(self):
        """Return a summary of stored entities."""
        collections = self.list_collections()
        collection_counts = {
            c: len(self.list_entities_by_collection(c)) for c in collections
        }
        return f"EntityManager({len(self.entities)} entities across {len(collections)} collections: {collection_counts})"
