from entity_graph.models.entity_graph.artifact import Artifact
from entity_graph.models.entity_graph.table import Table


def table_from_json(d):
    """
    Returns colum names and a list of rows
    If gets a dict of key: row, adds `id` column
    """
    cols = None
    rows = []
    if isinstance(d, dict):
        for k, v in d.items():
            if cols is None:
                cols = list(v.keys())
            rows.append({"id": k, **v})
        return ["id"] + cols, rows
    elif isinstance(d, list):
        return list(d[0].keys()), d


def add_table_from_json(
    new_manager,
    json_data,
    source_fn,
    table_name,
    row_key="id",
    data_type="instances",
    name_sep="_",
    collection="default",
):
    """
    json data is a list or dict, elements must be dicts
    source_fn is file name or File
    table_name is new table name
    row_key is row field used for artifact name, if not found autoincrements
    collection is the collection to which entities will be added (default: "default")
    """
    cols, rows = table_from_json(json_data)

    if isinstance(source_fn, str):
        src_file = new_manager.find_entity(source_fn, collection=collection)

    table = new_manager.find_entity(table_name, collection=collection)
    if table is None:
        table = Table(None, table_name, cols, {}, -1)
        # Set collection if the Table class supports it
        if hasattr(Table, "collection"):
            table.collection = collection
        new_manager.add_entity(table, collection=collection)
        table.add_relation(src_file, "in_file")
        src_file.add_relation(table, "in_file")

    for i, row in enumerate(rows):
        artifact = Artifact(
            None, table_name + name_sep + row.get(row_key, str(i)), row, {}, data_type
        )
        # Set collection if the Artifact class supports it
        if hasattr(Artifact, "collection"):
            artifact.collection = collection
        artifact.add_relation(table, "in_table")
        table.add_relation(artifact, "in_table")
        new_manager.add_entity(artifact, collection=collection)

    return table
