import pandas as pd
import pdfplumber


def parse_multiline_table(tab):
    """Takes first row as columns, assumes that the second row is a multiline string with rows"""
    return pd.DataFrame(zip(*map(lambda x: x.split("\n"), tab[1])), columns=tab[0])


def get_table(pdf_path, pages):
    """pages numbering starts from 1"""

    all_tables = {}
    # Open the PDF file with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if page_number not in pages:
                continue

            all_tables[page_number] = page.extract_tables()

    return all_tables


def parse_double_cols(table, header=True):
    n_cols = len(table[0])
    if n_cols % 2 == 1:
        raise
    half_cols = n_cols // 2

    return (
        [table[0][:half_cols]]
        + list(r[:half_cols] for r in table[1:])
        + list(r[half_cols:] for r in table[1:])
    )


def find_tables(filename, pages, header=None, test_func=None, is_duble_col=False):
    if test_func is None:
        test_func = lambda x: x[0] == header

    tabs = get_table(filename, pages)

    tables = {}
    for page in pages:

        tables[page] = list(t for t in tabs[page] if test_func(t))

        if is_duble_col:
            tables[page] = list(map(lambda x: parse_double_cols(x)[1:], tables[page]))

    return tables
