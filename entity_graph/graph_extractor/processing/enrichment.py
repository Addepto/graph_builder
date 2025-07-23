from entity_graph.graph_extractor.processing.model_structure import add_to_graph_copies
from entity_graph.models.entity_graph.object import Object


def flatten_desc(x):
    """
    Flatten a description that might be a list or other type

    :param x: Input value to flatten
    :return: Flattened string
    """
    if x is None:
        return ""

    if isinstance(x, list):
        # Filter out None values from list
        filtered_list = [item for item in x if item is not None]
        return "\n".join(str(item) for item in filtered_list)

    return str(x)


def simple_normalize(t):
    """
    Normalize a string by replacing newlines and multiple spaces

    :param t: Input string to normalize
    :return: Normalized string
    """
    if t is None:
        return ""

    return str(t).replace("\n", " ").replace("  ", " ")


def enrich_from_table(
    manager,
    table_rows,
    label1,
    label2,
    new_labels=None,
    relation="enrichment",
    debug=False,
    test=None,
    sep="|",
    join_sep=".",
    collection="default",
):
    """
    Checks all objects against given rows.
    If values under given match labels are equal, enriches the object

    Args:
        manager: Entity manager
        table_rows: Rows from table
        label1: Label to match in objects
        label2: Label to match in rows
        new_labels: Optional new labels for objects
        relation: Relation type
        debug: Whether to print debug info
        test: Test filter
        sep: Separator for names
        join_sep: Separator for joining values
        collection: Collection to use (default: "default")
    """
    # Create normalized values map from table rows
    normalized_values = {}
    for t in table_rows:
        # Skip if the label doesn't exist in the data
        if label2 not in t.data:
            print(f"Warning: Label '{label2}' not found in row data")
            continue

        # Skip if the value is None
        if t.data[label2] is None:
            print(f"Warning: Value for label '{label2}' is None")
            continue

        # Handle the value - convert to string and strip
        try:
            row_value = str(t.data[label2]).strip()
            if row_value:
                normalized_key = simple_normalize(t.data[label2])
                normalized_values[normalized_key] = normalized_values.get(
                    normalized_key, []
                ) + [t]
        except Exception as e:
            print(f"Error processing value for label '{label2}': {e}")
            continue

    # Get existing names in the collection
    existing_names = manager.refresh_name_map(collection=collection).keys()

    # Helper function to filter keys
    def filter_keys(d):
        return filter(lambda x: x != "id", d)

    # Process entities in the collection
    collection_entities = [
        e
        for e in manager.entities.values()
        if getattr(e, "collection", manager.default_collection) == collection
        and isinstance(e, Object)
    ]

    for o in set(collection_entities):
        # Skip if label1 doesn't exist in attributes
        if label1 not in o.attributes:
            continue

        # Skip if attribute value is None
        if o.attributes.get(label1) is None:
            continue

        new_child_names = set()

        # Get match value from object attributes - handle possible None
        try:
            match_value = simple_normalize(flatten_desc(o.attributes.get(label1, "")))
        except Exception as e:
            print(f"Error normalizing attribute '{label1}' for object '{o.name}': {e}")
            continue

        # Skip if no match
        if match_value not in normalized_values:
            continue

        # Get enrichments for this match
        enrichments = normalized_values[match_value]

        # Set confidence based on number of enrichments
        if len(enrichments) == 1:
            confidence = "high"
        else:
            confidence = "medium"

        # Process each enrichment
        for i, enrichment in enumerate(enrichments):
            parent = o

            # Create child name
            child_name = parent.name + sep + match_value

            # Use new labels if provided
            if new_labels is not None:
                # Safely get values for new labels
                new_name_values = []
                has_missing = False

                for c in new_labels:
                    if c not in enrichment.data:
                        print(f"Warning: Column '{c}' not found in enrichment data")
                        has_missing = True
                        break

                    value = enrichment.data.get(c)
                    if value is None or value == "None":
                        print(f"Warning: Value for column '{c}' is None")
                        has_missing = True
                        break

                    new_name_values.append(str(value))

                # Skip if missing values
                if has_missing:
                    print(
                        f"Skipping enrichment for {match_value} due to missing values"
                    )
                    continue

                child_name = parent.name + sep + join_sep.join(new_name_values)

            # Skip if name already exists
            if child_name in existing_names:
                print(f"Existing name: {child_name}")
                continue

            # Add suffix if duplicate in this enrichment
            if child_name in new_child_names:
                child_name += f"-{i}"

            # Add to set of new child names
            new_child_names.add(child_name)

            # Create new object
            obj = Object(None, child_name, parent.identifier_type, enrichment.data, {})

            # Set collection if supported
            if hasattr(obj, "collection"):
                obj.collection = collection

            # Add entity with collection
            manager.add_entity(obj, collection=collection)

            # Add relations
            manager.add_relation(enrichment.entity_id, obj.entity_id, "from_artifact")
            obj.add_relation(parent, "child_of", r_data={"confidence": confidence})
            parent.add_relation(obj, "parent_of", r_data={"confidence": confidence})

            if debug:
                if match_value.lower().startswith("ind"):
                    print(o.name, o.attributes[label1])
                elif match_value.lower().startswith("vari"):
                    print(o.name, o.attributes[label1])
                else:
                    print(o.name, o.attributes[label1])


# NOTE: hardcoded that from a table
def find_enrichment(new_manager, entity_name, collection="default"):
    """
    From an entity_name, finds the source table
    From table takes column identifiers
        (if identifier offers enrichment)
    Finds identifier key for this particular entity using entity attributes
    Returns enrichment sources available for this entity (its identifier key)

    Args:
        new_manager: Entity manager
        entity_name: Name of the entity
        collection: Collection to use (default: "default")
    Returns:
        List of enrichment sources
    """
    try:
        # Find entity in the specific collection
        entity = new_manager.named_entity(entity_name, collection=collection)
        if not entity:
            print(
                f"Warning: Entity '{entity_name}' not found in collection '{collection}'"
            )
            return []

        # Get from_artifact relations
        from_artifact_relations = entity.get_relations("from_artifact")
        if not from_artifact_relations or len(from_artifact_relations) == 0:
            return []

        # Get artifact
        artifact = new_manager.get_entity(from_artifact_relations[0])
        if not artifact:
            return []

        # Get in_table relations
        in_table_relations = artifact.get_relations("in_table")
        if not in_table_relations or len(in_table_relations) == 0:
            return []

        # Get table
        table = new_manager.get_entity(in_table_relations[0])
        if not table:
            return []

        # Check if table has identifiers
        if not hasattr(table, "identifiers") or not table.identifiers:
            print(f"Warning: Table '{table.name}' has no identifiers")
            return []

        # Get enrichments for identifiers
        enrichments = {}
        for identifier_id in table.identifiers:
            try:
                identifier = new_manager.get_entity(identifier_id)
                if not identifier:
                    continue

                enrichment = identifier.data.get("enrichment")
                if enrichment:
                    enrichments[identifier_id] = enrichment
            except Exception as e:
                print(f"Error processing identifier {identifier_id}: {e}")
                continue

        # Skip if no enrichments found
        if not enrichments:
            return []

        # Get enrichment keys
        enrichment_keys = {}
        for identifier_id in enrichments:
            try:
                # Get identifier columns
                identifier_cols = table.identifiers[identifier_id]
                if not identifier_cols or len(identifier_cols) == 0:
                    continue

                # Get entity attribute value for this identifier column
                col = identifier_cols[0]  # NOTE: only supports single column id
                if col in entity.attributes:
                    enrichment_keys[identifier_id] = entity.attributes[col]
            except Exception as e:
                print(f"Error getting key for identifier {identifier_id}: {e}")
                continue

        # Skip if no enrichment keys found
        if not enrichment_keys:
            return []

        # Get enrichment sources
        enrichment_sources = []
        for identifier_id, key in enrichment_keys.items():
            try:
                if key in enrichments[identifier_id]:
                    source = enrichments[identifier_id].get(key)
                    if source:
                        enrichment_sources.append(source)
            except Exception as e:
                print(f"Error getting source for key {key}: {e}")
                continue

        return [s for s in enrichment_sources if s is not None]
    except Exception as e:
        print(f"Error in find_enrichment for entity '{entity_name}': {e}")
        return []


def do_enrichment(manager, name, layout_id, collection="default"):
    """
    For a given entity name uses existing identifiers to find a link to a model definition.
    First look for enrichment sources,
    then makes a graph of the model structure
    then adds child nodes

    Args:
        manager: Entity manager
        name: Name of the entity
        layout_id: Layout identifier
        collection: Collection to use (default: "default")
    """
    # Refresh name map for the collection
    manager.refresh_name_map(collection=collection)

    # Find enrichment sources for the entity in this collection
    sources = find_enrichment(manager, name, collection=collection)
    if not sources:
        return

    src_table = manager.get_entity(sources[0])
    if not src_table:
        print(
            f"Warning: Source table not found for entity '{name}' in collection '{collection}'"
        )
        return

    # Safety check for columns
    if not hasattr(src_table, "columns") or not src_table.columns:
        print(f"Warning: Source table '{src_table.name}' has no columns")
        return

    if len(src_table.columns) == 0:
        print(f"Warning: Source table '{src_table.name}' has empty columns list")
        return

    # Get table contents for this collection
    try:
        model_data = manager.get_table_contents_dict(
            src_table, src_table.columns[0], collection=collection
        )
    except Exception as e:
        print(f"Error getting table contents for '{src_table.name}': {e}")
        return

    # Add to graph
    try:
        nodes, edges = add_to_graph_copies(
            name,
            model_data,
        )
    except Exception as e:
        print(f"Error adding to graph copies for '{name}': {e}")
        return

    # Ensure values is a set
    if "values" not in layout_id.data:
        layout_id.data["values"] = set()
    elif isinstance(layout_id.data["values"], list):
        layout_id.data["values"] = set(layout_id.data["values"])

    for node in nodes:
        try:
            node_id, node_attributes = node
            layout_id.data["values"].add(node_id)
            node_attributes["created_from_model_enrichment"] = True
            node_attributes["from_model"] = src_table.name

            # Create new object
            obj = Object(None, node_id, layout_id.entity_id, node_attributes, {})

            # Set collection if supported
            if hasattr(obj, "collection"):
                obj.collection = collection

            # Add entity with collection
            manager.add_entity(obj, collection=collection)

            artf_id = node_attributes.get("source_entity")
            if artf_id:
                manager.add_relation(artf_id, obj.entity_id, "from_artifact")
        except Exception as e:
            print(f"Error processing node: {e}")
            continue

    # Refresh name map for the collection
    manager.refresh_name_map(collection=collection)

    for edge in edges:
        try:
            n1, n2 = edge
            # Find entities in the specific collection
            n1_entity = manager.named_entity(n1, collection=collection)
            n2_entity = manager.named_entity(n2, collection=collection)

            if n1_entity and n2_entity:
                n1_entity.add_relation(n2_entity, "child_of")
                n2_entity.add_relation(n1_entity, "parent_of")
            else:
                if not n1_entity:
                    print(f"Entity '{n1}' not found in collection '{collection}'")
                if not n2_entity:
                    print(f"Entity '{n2}' not found in collection '{collection}'")
        except Exception as e:
            print(f"Error processing edge: {e}")
            continue


def enrich_identifier_values(manager, id_name, collection="default"):
    """
    Enrich all values in the identifier with their respective models.

    Args:
        manager: Entity manager
        id_name: Name of the identifier to enrich
        collection: Collection to use (default: "default")
    """
    layout_id = manager.find_entity(id_name, collection=collection)
    if not layout_id:
        print(f"Warning: Identifier '{id_name}' not found in collection '{collection}'")
        return

    if "values" not in layout_id.data:
        layout_id.data["values"] = set()
    elif isinstance(layout_id.data["values"], list):
        layout_id.data["values"] = set(layout_id.data["values"])

    for name in list(layout_id.data["values"]):
        do_enrichment(manager, name, layout_id, collection=collection)


def enrich_from_data_with_map(
    new_manager, obj, label, map_data, enrichment_data, collection="default"
):
    """
    Takes `enrichment_data` and `map_data`.
    Looks at label in the object and uses a map to find a 'synonym'
    Uses `enrichment_data` to get actual data, including source entity_id

    Args:
        new_manager: Entity manager
        obj: Object to enrich
        label: Label to check in object
        map_data: Mapping data
        enrichment_data: Enrichment data
        collection: Collection to use (default: "default")
    """
    if obj.attributes.get(label) in map_data:
        art_id = map_data[obj.attributes[label]]
        artifact_data = enrichment_data[art_id]
        art_eid = artifact_data["source_entity"]

        for k in artifact_data:
            if k not in obj.attributes:
                obj.attributes[k] = artifact_data[k]
        new_manager.add_relation(obj.entity_id, art_eid, "enrichment")


def enrich_from_table_name(
    new_manager, table_name, label1, label2, new_labels=None, collection="default"
):
    """
    Adds new objects as children of existing objects if there is a match.
    Compares `label1` in existing object with `label2` from a row.
    The new name is the match or values of `new_labels`.

    Args:
        new_manager: Entity manager
        table_name: Name of the table
        label1: Label to compare in existing objects
        label2: Label to compare in table rows
        new_labels: Optional new labels to use for new objects
        collection: Collection to use (default: "default")
    """
    enrich_table = new_manager.find_entity(table_name, collection=collection)
    if not enrich_table:
        print(f"Warning: Table '{table_name}' not found in collection '{collection}'")
        return

    table_rows = [new_manager.get_entity(i) for i in enrich_table.get_rows_ids()]

    enrich_from_table(
        new_manager, table_rows, label1, label2, new_labels, collection=collection
    )


def add_enrichment_links(new_manager, id_name, attr_name, key, collection="default"):
    """
    Assign enrichment tables/artifacts for identifier values.
    If an entity attribute is a dict with a key holding value matching id, store reference.

    Args:
        new_manager: Entity manager
        id_name: Name of the identifier
        attr_name: Attribute name to check
        key: Key in the attribute to match against values
        collection: Collection to use (default: "default")
    """
    id_obj = new_manager.find_entity(id_name, collection=collection)
    if not id_obj:
        print(f"Warning: Identifier '{id_name}' not found in collection '{collection}'")
        return

    if "values" not in id_obj.data:
        id_obj.data["values"] = set()
    elif isinstance(id_obj.data["values"], list):
        id_obj.data["values"] = set(id_obj.data["values"])

    values = id_obj.data["values"]

    if id_obj.data.get("enrichment") is None:
        id_obj.data["enrichment"] = {}

    collection_entities = [
        e
        for e in new_manager.entities.values()
        if getattr(e, "collection", new_manager.default_collection) == collection
    ]

    for e in collection_entities:
        try:
            v = getattr(e, attr_name)[key]
        except:
            v = None
        if v is not None and v in values:
            id_obj.data["enrichment"][v] = e.entity_id
            e.add_relation(id_obj, "enrichment")
            id_obj.add_relation(e, "enrichment")
            if attr_name == "header":
                e.header_identifiers[id_obj.entity_id] = [key]
