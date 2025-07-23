from entity_graph.graph_extractor.processing.hierarchy import add_hierarchy
from entity_graph.graph_manager import EntityManager
from entity_graph.models.entity_graph.artifact import Artifact
from entity_graph.models.entity_graph.identifier import Identifier
from entity_graph.models.entity_graph.object import Object
from entity_graph.models.entity_graph.table import Table

AUTOINREMENT = "autoincrement"


def normalize_identifier_value(value):
    return value.replace(".", "")


def make_objects(
    manager: EntityManager,
    eplan_id: Artifact | None,
    eplan_table: Table,
    override_cols=None,
    collection="default",
):
    """
    Creates objects from table using an identifier

    :param manager: Entity manager
    :param eplan_id: Identifier artifact
    :param eplan_table: Table to create objects from
    :param override_cols: Optional column overrides
    :param collection: Collection to use (default: "default")
    """
    # TODO: support complex indentifiers
    identifier_cols = eplan_table.identifiers[eplan_id.entity_id]
    if override_cols:
        identifier_cols = override_cols

    def get_id(row_data, i, sep="-"):
        # TODO: check if missing is AUTOINREMENT?
        values = [
            str(row_data.get(col, i if col == AUTOINREMENT else None))
            for col in identifier_cols
        ]
        if None in values:
            return
        if len(eplan_table.identifiers[eplan_id.entity_id]) == 1:
            return values[0]
        else:
            return sep.join(values)

    # Get rows for this specific collection
    rows = manager.get_table_rows(eplan_table, collection=collection)
    for i, row in enumerate(rows):
        row_identifier = get_id(row.data, i)
        if row_identifier is None:
            continue
        if row_identifier == "None":
            continue
        if row_identifier == "":
            continue
        row_identifier = normalize_identifier_value(row_identifier)

        # print("row_identifier", row_identifier)
        if row_identifier not in eplan_id.data["values"]:
            eplan_id.data["values"].add(row_identifier)
            obj = Object(None, row_identifier, eplan_id.entity_id, row.data, {})

            # Set collection if supported
            if hasattr(obj, "collection"):
                obj.collection = collection

            manager.add_entity(obj, collection=collection)
            row.add_relation(obj, "from_artifact")
            obj.add_relation(row, "from_artifact")
        else:
            # TODO: merge metadata
            # TODO: logger
            print(
                f"WARNING: duplicated identifier {row_identifier} in collection {collection}"
            )


def create_instances(
    manager, id_name, table_name, do_hierarchy, override_cols=None, collection="default"
):
    """
    Create instances using the specified identifier and table

    Args:
        manager: Entity manager
        id_name: Name of the identifier
        table_name: Name of the table
        do_hierarchy: Whether to create hierarchy relationships
        override_cols: Optional column overrides
        collection: Collection to use (default: "default")
    """
    eplan_id = manager.find_entity(id_name, collection=collection)
    if not eplan_id:
        print(f"Warning: Identifier '{id_name}' not found in collection '{collection}'")
        return

    eplan_table = manager.find_entity(table_name, collection=collection)
    if not eplan_table:
        print(f"Warning: Table '{table_name}' not found in collection '{collection}'")
        return

    if "values" not in eplan_id.data:
        eplan_id.data["values"] = set()
    elif isinstance(eplan_id.data["values"], list):
        eplan_id.data["values"] = set(eplan_id.data["values"])

    make_objects(
        manager,
        eplan_id,
        eplan_table,
        override_cols=override_cols,
        collection=collection,
    )

    if do_hierarchy:
        add_hierarchy(manager, eplan_id, collection=collection)


def new_id(new_manager, id_name, table_name, cols, fill_values, collection="default"):
    """
    Create a new id for a table columns selection

    :param new_manager: Entity manager
    :param id_name: Name of the identifier
    :param table_name: Name of the table
    :param cols: Columns to use for the identifier
    :param fill_values: Whether to fill values from the table
    :param collection: Collection to which entities will be added (default: "default")
    :return: Created identifier
    """
    eplan_id = Identifier(None, id_name, {})
    eplan_id.data["values"] = set()

    # Set collection if supported
    if hasattr(eplan_id, "collection"):
        eplan_id.collection = collection

    # Add the entity with collection
    new_manager.add_entity(eplan_id, collection=collection)

    # Find table in the specific collection
    eplan_table = new_manager.find_entity(table_name, collection=collection)
    if not eplan_table:
        print(f"Warning: Table '{table_name}' not found in collection '{collection}'")
        return eplan_id

    eplan_table.identifiers[eplan_id.entity_id] = cols

    if fill_values:
        if len(cols) == 1:
            # Get values from table rows for this collection
            table_rows = new_manager.get_table_rows(eplan_table, collection=collection)
            values = set()

            for row in table_rows:
                if cols[0] in row.data:
                    values.add(row.data[cols[0]])

            eplan_id.data["values"] = values
        else:
            raise NotImplementedError

    return eplan_id


def link_id(new_manager, id_name, table_name, cols, fill_values, collection="default"):
    """
    Create a new id for a table columns selection

    :param new_manager: Entity manager
    :param id_name: Name of the identifier
    :param table_name: Name of the table
    :param cols: Columns to use for the identifier
    :param fill_values: Whether to fill values from the table
    :param collection: Collection to which entities will be added (default: "default")
    :return: Updated identifier
    """
    # Find identifier in the specific collection
    eplan_id = new_manager.find_entity(id_name, collection=collection)
    if not eplan_id:
        print(f"Warning: Identifier '{id_name}' not found in collection '{collection}'")
        return None

    # Find table in the specific collection
    eplan_table = new_manager.find_entity(table_name, collection=collection)
    if not eplan_table:
        print(f"Warning: Table '{table_name}' not found in collection '{collection}'")
        return eplan_id

    eplan_table.identifiers[eplan_id.entity_id] = cols

    # Ensure values is a set
    if "values" not in eplan_id.data:
        eplan_id.data["values"] = set()
    elif isinstance(eplan_id.data["values"], list):
        eplan_id.data["values"] = set(eplan_id.data["values"])

    if fill_values:
        if len(cols) == 1:
            # Get values from table rows for this collection
            table_rows = new_manager.get_table_rows(eplan_table, collection=collection)
            new_values = set()

            for row in table_rows:
                if cols[0] in row.data:
                    new_values.add(row.data[cols[0]])

            # Use update() instead of union_update
            eplan_id.data["values"].update(new_values)
        else:
            raise NotImplementedError

    return eplan_id
