import re

import fitz
from pydantic import BaseModel

from entity_graph.graph_extractor.processing.extraction_page import (
    PageComponents,
    read_page,
)

r"""
Approach:
- read all text blocks
- find matching row identifiers
- join column blocks into rows using y position similarity
- parse rows/lines, including splitting on \t as some cells may be returned as one block

"""


def row_test(txt):
    return txt.startswith("++") and len(txt) > 2


def detect_rows(page_components: PageComponents):
    return [
        i for i, (txt, pos) in enumerate(page_components.get_blocks()) if row_test(txt)
    ]


def get_page_tables(fname, page, max_dist=200):
    """
    finds rows, return id, position and all blocks
    """
    page_components = read_page(fname, page)
    block_positions = page_components.text_positions

    row_starts = detect_rows(page_components)

    # x positions
    main_columns = [page_components.text_positions[i][0] for i in row_starts]

    # indexing blocks by y position
    lines = {y: [] for y in set([int(p[1]) for p in block_positions])}
    for txt, pos in page_components.get_blocks():
        lines[int(pos[1])].append((txt, [int(y) for y in pos]))

    # for all x positions matching columns
    # find close y blocks within max_dist in y direction
    main_entries = [
        (
            txt,
            int(pos[0]),
            int(pos[1]),
            [l for l in lines[int(pos[1])] if abs(pos[0] - l[1][0]) < max_dist]
            + [
                l
                for l in lines.get(int(pos[1]) - 1, [])
                if abs(pos[0] - l[1][0]) < max_dist
            ]
            + [
                l
                for l in lines.get(int(pos[1]) + 1, [])
                if abs(pos[0] - l[1][0]) < max_dist
            ],
        )
        for txt, pos in page_components.get_blocks()
        if pos[0] in main_columns
    ]
    return main_entries


def get_multiline_tables_raw(config, debug=False):
    """
    Finds rows text.
    Find row blocks and extract text.
    """
    fname = config.filename
    pages_range = config.pages
    extracts = []
    for page_num in range(pages_range[0], pages_range[1]):
        # get rows with positions
        all_rows = get_page_tables(fname, page_num, max_dist=200)
        if debug:
            print(f"page: {page_num}: {len(all_rows)}")
        for row_data in all_rows:
            # list of text values sorted by x position
            row = list(
                map(
                    lambda x: x[0],
                    sorted(
                        row_data[3], key=lambda x: x[1][0]
                    ),  # row blocks, sorted by x
                )
            )
            extracts.append(row)
    return extracts


class LineParser:
    @staticmethod
    def get_row_label(row):
        elements = []
        for r in row:
            # we join by \t in read_page
            elements += r.split("\t")
        return list(map(lambda x: x.strip(), elements))

    @staticmethod
    def fix_metadata(l):
        d = {
            "Artikelbezeichnung": [],
            "QVW": [],
            # "Unknown": [],
        }
        for el in l:
            if el.startswith("++"):
                d["QVW"].append(el)
            else:
                d["Artikelbezeichnung"].append(el)
        for k in d:
            if len(d[k]) == 1:
                d[k] = d[k][0]
        return d

    @staticmethod
    def parse(row):
        pattern_prefix = r"\+\+.....\+.....[^\&]*$"
        pattern_leaf = r"-.*"
        elements = LineParser.get_row_label(row)
        return {
            "prefix": [el for el in elements if re.match(pattern_prefix, el)],
            "leaf": [el for el in elements if re.match(pattern_leaf, el)],
            "metadata": LineParser.fix_metadata(
                [
                    el
                    for el in elements
                    if not any(
                        [
                            re.match(pattern, el)
                            for pattern in [pattern_leaf, pattern_prefix]
                        ]
                    )
                ]
            ),
        }

    @staticmethod
    def get_multiline_tables_parsed(raw_data, debug=False):
        parsed = [LineParser.parse(row) for row in raw_data]
        return parsed


def get_leaf(p, k="leaf"):
    if len(p[k]) > 0:
        return p[k][0]
    return ""


class ExtractionMultiline:

    @staticmethod
    def get_multiline_tables(config, debug=False):
        prefixes = config.prefixes

        raw_data = get_multiline_tables_raw(config, debug=debug)
        parsed = LineParser.get_multiline_tables_parsed(raw_data, debug=debug)

        # TODO:  we should not make a dict just to convert to a list after extraction
        as_dict = {}
        for parsed_entry in parsed:
            idx = get_leaf(parsed_entry, k="prefix") + get_leaf(parsed_entry)
            if prefixes:
                for column_name, prefix in prefixes:
                    if column_name == "id" and idx.strip():
                        idx = prefix + idx
                    elif (
                        column_name in parsed_entry
                        and parsed_entry[column_name].strip()
                    ):
                        parsed_entry[column_name] = prefix + parsed_entry[column_name]
            if idx and idx in as_dict:
                as_dict[idx] = {**as_dict[idx], **parsed_entry["metadata"]}
            if idx and idx not in as_dict:
                as_dict[idx] = parsed_entry["metadata"]

        return as_dict
