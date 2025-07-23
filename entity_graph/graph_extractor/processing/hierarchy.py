import re

from entity_graph.models.entity_graph.object import Object

LABEL_NORMALIZATION_SEPEARATOR = "-"
MATCH_CHARS = r"[\s|_|/|\-|\+|=]"


def replace_multiple_spaces(text, match=r"\s", put=" "):
    """
    Replaces any sequence of spaces in the input text with a single space.

    Args:
        text (str): The input string.
        match (str): Regex pattern for characters to replace
        put (str): Replacement string

    Returns:
        str: The modified string with matched patterns replaced
    """
    if type(text) != str:
        return f"None"
    return re.sub(match + "+", put, text)


def normalize(l):
    """
    Normalize a label by replacing matched characters with separators

    :param l: Label to normalize
    :return: Normalized label
    """
    p1 = replace_multiple_spaces(
        l, match=r"[\s|_|/|\-|\+|=]", put=LABEL_NORMALIZATION_SEPEARATOR
    )
    return p1


def get_separator_hierarchy(values):
    """
    Process values into a hierarchy structure based on separators

    :param values: Values to process
    :return: Tuple of forward and reverse maps
    """
    map1 = {
        v: tuple(p for p in normalize(v).split(LABEL_NORMALIZATION_SEPEARATOR) if p)
        for v in values
    }
    map2 = {v: k for k, v in map1.items()}
    return map1, map2


def get_indirect_nodes(kmap, rmap):
    """
    Tries to find all parent nodes that do not appear explicitly.
    For each name in the input dict checks all possible parent nodes.
    Extends given dicts and returns the list of added nodes.

    :param kmap: Key to decomposition map
    :param rmap: Decomposition to key map
    :return: Set of indirect nodes
    """
    seps = MATCH_CHARS + "*"
    indirect_nodes = set()
    for key, decomposition in list(kmap.items()):
        if len(decomposition) <= 1:
            continue
        n_parents = len(decomposition) - 1
        for i in range(1, 1 + n_parents):
            parent = decomposition[:-i]

            if parent not in rmap:
                try:
                    parent_name = re.match(seps + seps.join(parent), key).group()
                except:
                    print("error", key, parent)
                    raise
                indirect_nodes.add(parent_name)
                rmap[parent] = parent_name
                kmap[parent_name] = parent
    return indirect_nodes


def get_separator_hierarchy_edges(values):
    """
    Use separator normalization to split ids and extract hierarchy.
    All edges are (child, parent).

    :param values: Values to extract hierarchy from
    :return: Tuple of edges and indirect nodes
    """
    kmap, rmap = get_separator_hierarchy(values)
    edges = set()
    indirect_nodes = get_indirect_nodes(kmap, rmap)
    for v in kmap:
        t = kmap[v]
        parent = t[:-1]
        if len(t) <= 1:
            continue
        if parent in rmap:
            edges.add((v, rmap[parent]))
        else:
            raise ValueError("All parent should be already added")
    return edges, indirect_nodes


def add_hierarchy(new_manager, eplan_id, collection="default"):
    """
    Add hierarchy relationships based on identifier values

    :param new_manager: Entity manager
    :param eplan_id: Identifier to use for hierarchy
    :param collection: Collection to use (default: "default")
    :return: None
    """
    eplan_edges, indirect_nodes = get_separator_hierarchy_edges(eplan_id.data["values"])

    # Create intermediate nodes that aren't explicitly defined
    for node in indirect_nodes:
        obj = Object(None, node, eplan_id.entity_id, {}, {"derived": True})

        # Set collection if supported
        if hasattr(obj, "collection"):
            obj.collection = collection

        eplan_id.data["values"].add(node)

        # Add entity with collection
        new_manager.add_entity(obj, collection=collection)

    # Refresh name map for the specific collection
    new_manager.refresh_name_map(collection=collection)

    # Add parent-child relationships
    for n1, n2 in eplan_edges:
        # Find entities in the specific collection
        o1 = new_manager.named_entity(n1, collection=collection)
        o2 = new_manager.named_entity(n2, collection=collection)

        if o1 and o2:
            o1.add_relation(o2, "child_of")
            o2.add_relation(o1, "parent_of")
        else:
            if not o1:
                print(f"Entity '{n1}' not found in collection '{collection}'")
            if not o2:
                print(f"Entity '{n2}' not found in collection '{collection}'")
