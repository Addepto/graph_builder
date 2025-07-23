class ModelTypes:
    MECHANICAL_MODEL = "mechanical_model"


def mechanical_model_parsing(): ...


def get_model_copies(
    model_data, count_index="Stück Unit", sep=".", instance_sep="-", group_sep="/"
):
    """
    From model data get a list of subcomponents, including copies and copy id.
    Returns a list (subpart_id, copy_id, parent_suffix, node_suffix)
    """
    counts = {k: int(v[count_index]) for k, v in model_data.items()}
    levels = {}
    for k in model_data:
        l = k.count(sep)
        levels[l] = levels.get(l, []) + [k]

    new_labels = []
    for level in levels:
        for k in levels[level]:
            parent_key = None if level == 0 else sep.join(k.split(sep)[:-1])
            parents = (
                [(None, [])]
                if level == 0
                else list(
                    filter(lambda x: sep.join(k.split(sep)[:-1]) == x[0], new_labels)
                )
            )
            for parent in parents:
                j = parent[1]
                for i in range(counts[k]):
                    new_ids = j + [str(i)]
                    parent_label = (
                        ""
                        if parent_key is None
                        else f"{parent_key}{group_sep}{instance_sep.join(j)}"
                    )
                    new_labels.append(
                        (
                            k,
                            new_ids,
                            parent_label,
                            f"{k}{group_sep}{instance_sep.join(new_ids)}",
                        )
                    )
                    # new_labels.append((k, new_ids, f"{parent_key}/{'-'.join(j)}"), f"{k}/{'-'.join(new_ids)}"))
    return new_labels


get_idx_label = lambda idx: str(idx + 1)


def add_to_graph_copies(
    model_root_label,
    model_data,
    include_root=False,
    root_sep="-",
    count_index="Stück Unit",
    sep=".",
    instance_sep="-",
    group_sep="/",
):
    """
    Creates a list of instances based on model data, starts from model root (if `include_root`)
    names follow pattern: {model_root_label}-{pos}/{instance_id}
    where {instance_id} is a list of instance ids in hierarchy separated with '-'

    all edges should be (child, parent)
    """

    subpart_copies = get_model_copies(
        model_data,
        count_index=count_index,
        sep=sep,
        instance_sep=instance_sep,
        group_sep=group_sep,
    )
    nodes = []
    edges = []

    def add_node(*args, **kwargs):
        nodes.append((args[0], kwargs))

    def add_edge(*args, **kwargs):
        edges.append(args)

    if include_root:
        add_node(
            model_root_label,
            mechanical=True,
        )

    # Iterate instances, join with model data
    for subpart_data in subpart_copies:
        pos, copy_ids, parent_id, node_id = subpart_data
        subpart = model_data[pos]
        subpart_label = root_sep.join((model_root_label, node_id))
        if parent_id:
            edge = root_sep.join((model_root_label, node_id)), root_sep.join(
                (model_root_label, parent_id)
            )
        else:
            edge = root_sep.join((model_root_label, node_id)), model_root_label
        add_node(subpart_label, mechanical=True, **subpart)
        add_edge(*edge)
    return nodes, edges


def add_instances_info(new_manager, link_config, info_source, collection=None):

    instance_sep = "/"
    submodel_sep = "-"
    attr_sep = "."
    submodel_attr = "Pos."
    instances_attr = "Stück Unit"
    info_key = "step"

    id_name = link_config[0]
    target_model = link_config[1]

    identifier = new_manager.find_entity(id_name)

    for entity in new_manager.entities.values():
        if entity.entity_type != "object":
            continue

        if "from_model" not in entity.attributes:
            continue

        model = entity.attributes["from_model"].split("model_")[1]

        if model != target_model:
            continue

        if model not in identifier.data["values"]:
            continue

        name, pos = entity.name, str(entity.attributes.get(submodel_attr, ""))
        node_meta = entity.attributes
        posand, instance = name.split(instance_sep)
        pos = posand.split(submodel_sep)[-1]

        step_data = info_source.get(pos, {}).get(info_key, {})
        if attr_sep in pos and pos.count(attr_sep) == 1:
            parent = pos.split(attr_sep)[0]
            step_data = (
                info_source.get(parent, {})
                .get("children", {})
                .get(pos, {})
                .get(info_key, {})
            )
            instance = instance.split(submodel_sep)[-1]
        elif pos.count(attr_sep) > 1:
            parent1 = pos.split(attr_sep)[0]
            parent2 = pos.split(attr_sep)[1]
            sp1 = info_source.get(parent1, {}).get("children", {})
            sp2 = sp1.get(parent1 + attr_sep + parent2, {}).get("children", {})
            step_data = sp2.get(pos, {}).get(info_key, {})
            instance = instance.split(submodel_sep)[-1]
        if isinstance(step_data, list):
            if len(step_data) > 0:
                pieces = int(node_meta.get(instances_attr, 0))
                if len(step_data) < pieces:
                    ...
                i = int(instance)
                if i < len(step_data):
                    step_data = step_data[i]
                else:
                    step_data = {}
            else:
                step_data = {}
        entity.attributes["instance_info"] = step_data


def match_method(e, model_attr):
    return e.attributes[model_attr]


def station_filter(name):
    try:
        return name.split("-")[-1][0] in ("T", "M")
    except Exception as e:
        return False


def station_key(name):
    return name.split("-")[-1]


def find_siblings(new_manager, name, parent_attribute, _filter=None, _key_map=None):
    matching_parent = new_manager.named_entity(name).attributes[parent_attribute]
    matching_parent = new_manager.named_entity(matching_parent)
    if matching_parent is None:
        return {}
    if _filter is None:
        _filter = lambda x: True
    if _key_map is None:
        _key_map = lambda x: x
    return {
        _key_map(new_manager.get_entity(eid).name): eid
        for eid in matching_parent.get_relations("parent_of")
        if _filter(new_manager.get_entity(eid).name)
    }


def get_all_child_nodes(new_manager, n):
    """Tree traversal for children"""
    node = new_manager.find_entity(n)
    children = node.get_relations("parent_of")
    todo = node.get_relations("parent_of").copy()
    visited = set(todo)

    while todo:
        cid = todo.pop()
        new_children = new_manager.get_entity(cid).get_relations("parent_of")
        children += new_children

        for child in new_children:
            if child not in visited:
                visited.add(child)
                todo.append(child)

    return children


def find_label_value(new_manager, l, label, value):
    """
    For all entity_ids in `l` takes those that have attribute `label`==`value`.
    Returns (entity_id, attributes)
    """
    return [
        (e, new_manager.get_entity(e).attributes)
        for e in l
        if new_manager.get_entity(e).attributes[label] == value
    ]
