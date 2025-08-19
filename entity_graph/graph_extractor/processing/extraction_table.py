import re
from collections import Counter

import numpy as np
import pandas as pd
import pdfplumber
from pydantic import BaseModel

from entity_graph.graph_extractor.processing.config_factory import config_factory
from entity_graph.graph_extractor.processing.extraction_multiline import (
    ExtractionMultiline,
)
from entity_graph.graph_extractor.processing.extraction_page import find_label_info
from entity_graph.graph_extractor.processing.get_table import (
    find_tables,
    parse_double_cols,
)
from entity_graph.graph_extractor.processing.models import (
    TableFromHeaderExtracionConfig,
    TableWithContextExtracionConfig,
    TableXlsExtracionConfig,
)
from entity_graph.graph_extractor.processing.utils import (
    get_data,
    replace_multiple_spaces,
)

"""
Contains:
"""


def header_condition_factory(header):
    return lambda t: t[0] == header


def split_column(df, column_name, sep):
    """
    Splits a column in a DataFrame into new columns based on a separator.

    Args:
    - df (pd.DataFrame): The DataFrame containing the column to split.
    - column_name (str): The name of the column to split.
    - sep (str): The separator used to split the column values.

    Returns:
    - pd.DataFrame: The modified DataFrame with new columns.
    """
    new_names = column_name.split(sep)
    new_columns = pd.DataFrame(columns=column_name.split(sep))
    for i, row in df.iterrows():
        new_values = row[column_name].split(sep)
        new_columns.loc[i] = new_values[: len(new_names)] + [""] * max(
            0, len(new_names) - len(new_values)
        )
    return pd.concat([df, new_columns], axis=1)


def get_all_matching_tables(
    config: TableFromHeaderExtracionConfig,
    debug: bool = False, 
):
    """
    Extracts all tables from a PDF file that match a given condition.

    Args:
        pdf_path (str): The path to the PDF file.
        table_test (function): A function that takes a table as input and returns a boolean indicating whether the table matches the desired condition.
        page_range (list, optional): A list of page numbers to extract tables from. If None, extracts tables from all pages.

    Returns:
        list: A list of tables that match the given condition.

    Notes:
        This function uses pdfplumber to extract tables from the PDF file.
    """
    oktables = []
    pdf_path = config.filename
    prefixes = config.prefixes
    columns_to_split = config.columns_to_split
    table_test = header_condition_factory(config.header)
    page_range = range(config.pages[0], config.pages[1]) if config.pages else None

    # Open the PDF file with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if page_range and page_number not in page_range:
                continue
            print("extract", page_number)
            tables = page.extract_tables()
            for table_number, table in enumerate(tables, start=1):
                if table_test(table):
                    oktables.append(table)
                elif debug:
                    print("not ok", table[0])

    if debug:
        return oktables

    all_elements = pd.concat(
        [pd.DataFrame(okt[1:], columns=okt[0]) for okt in oktables], ignore_index=True
    )

    if columns_to_split:
        for column_name, sep in columns_to_split:
            if column_name in all_elements.columns:
                all_elements = split_column(all_elements, column_name, sep)

    if prefixes:
        for column_name, prefix in prefixes:
            if column_name in all_elements.columns:
                all_elements[column_name] = all_elements[column_name].map(
                    lambda x: prefix + x if x != "" else ""
                )

    return all_elements


################ multiline


def get_multiline_tables(config):
    return ExtractionMultiline.get_multiline_tables(config)


################ with context


def get_tables_with_context(config: TableWithContextExtracionConfig):
    filename = config.filename
    pages = range(config.pages[0], config.pages[1])
    label = config.context_label
    header_double = config.header_double
    header_single = config.header_single
    header_output = config.header_output
    prefixes = config.prefixes

    if not header_single and not header_double:
        raise ValueError
    if not header_single and not header_output:
        raise ValueError
    if not header_output:
        header_output = header_single

    def table_test(t):
        # NOTE: we assume t[0] is never None
        return t[0] == header_single or t[0] == header_double

    def map_table(t):
        if t[0] == header_single:
            return t
        return parse_double_cols(t)

    raw_tables = find_tables(filename, pages, test_func=table_test)

    non_empty_pages = [k for k, v in raw_tables.items() if v]
    page_context = {}
    for page_to_extract in non_empty_pages:
        try:
            groups = find_label_info(
                filename, page_number=int(page_to_extract), label=label
            )
            page_context[page_to_extract] = "".join(
                groups[0].split(" ") + groups[1].split(" ")
            )
        except Exception as e:
            # TODO LOG and will raise KeyError in the next loop anyway
            print(f"failed {page_to_extract}")
            print(f"failed {e}")

    all_rows = []
    for page in non_empty_pages:
        table = map_table(raw_tables[page][0])
        for row in table[1:]:
            row[1] = page_context[page] + row[1] if row[1] else row[1]
        all_rows += table[1:]
    parts_df = pd.DataFrame(all_rows, columns=header_output)

    if prefixes:
        for column_name, prefix in prefixes:
            if column_name in parts_df.columns:
                parts_df[column_name] = parts_df[column_name].map(
                    lambda x: prefix + x if x is not None and x != "" else ""
                )

    return parts_df


def get_excel_table(config: TableXlsExtracionConfig):
    data_path = config.filename
    labels = config.labels_row_id
    data_start = config.data_start_row_id
    required = config.required_field
    prefixes = config.prefixes
    normalize = config.normalize
    protected = config.protected

    # TODO: some config is optional
    if protected:
        try:
            df = get_data(data_path)
        except Exception as e:
            print(data_path)
            raise e
    else:
        df = pd.read_excel(data_path, header=None)
    # cols = df.iloc[labels].to_list()
    df.columns = list(
        map(replace_multiple_spaces, df.iloc[labels].to_list())
    )  # no index
    df = df.iloc[data_start:]
    # df.columns = cols
    if required:
        df = df.loc[~df[required].isna()]
    if prefixes:
        for column_name, prefix in prefixes:
            if column_name in df.columns:
                df[column_name] = df[column_name].map(
                    lambda x: prefix + x if x != "" else ""
                )
    if normalize:
        for kind, column_name, x, y in normalize:
            if kind == "replace":
                df[column_name] = df[column_name].map(lambda t: t.replace(x, y))
    return df
