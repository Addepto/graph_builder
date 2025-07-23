# based on: https://stackoverflow.com/a/52290873/9560908

import contextlib
import json
import re
import tempfile
from pathlib import Path

import msoffcrypto
import pandas as pd
import xlrd

LABEL_NORMALIZATION_SEPEARATOR = "-"


@contextlib.contextmanager
def handle_protected_workbook(wb_filepath):
    try:
        with xlrd.open_workbook(wb_filepath) as wb:
            yield wb
    except xlrd.biffh.XLRDError as e:
        if str(e) != "Workbook is encrypted":
            raise
        # Try and unencrypt workbook with magic password
        wb_path = Path(wb_filepath)
        with wb_path.open("rb") as fp:
            wb_msoffcrypto_file = msoffcrypto.OfficeFile(fp)
            try:
                # Yes, this is actually a thing
                # https://nakedsecurity.sophos.com/2013/04/11/password-excel-velvet-sweatshop/
                wb_msoffcrypto_file.load_key(password="VelvetSweatshop")
            except Exception as e:
                raise Exception('Unable to read file "{}"'.format(wb_path)) from e

            # Magic Excel password worked
            with tempfile.NamedTemporaryFile(delete=False) as tmp_wb_unencrypted_file:
                # Decrypt into the tempfile
                wb_msoffcrypto_file.decrypt(tmp_wb_unencrypted_file)
                decrypted_path = Path(tmp_wb_unencrypted_file.name)
            try:
                with xlrd.open_workbook(str(decrypted_path)) as wb:
                    yield wb
            finally:
                decrypted_path.unlink()


def replace_multiple_spaces(text, match=r"\s", put=" "):
    """
    Replaces any sequence of spaces in the input text with a single space.

    Args:
        text (str): The input string.

    Returns:
        str: The modified string with consecutive spaces replaced by a single space.
    """
    if type(text) != str:
        return f"None"
    return re.sub(match + "+", put, text)


def get_data(filepath):
    try:
        with handle_protected_workbook(filepath) as wb:
            stucklisten = pd.read_excel(wb, header=None)
    except:
        stucklisten = pd.read_csv(filepath, header=None)

    return stucklisten


def read_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d


def table_from_pandas(df):
    """
    Read pandas like json
    """
    return [row.to_dict() for i, row in df.iterrows()]


def fix_na(col):
    if isinstance(col, list):
        return [fix_na(e) for e in col]
    if isinstance(col, dict):
        return {k: fix_na(e) for k, e in col.items()}
    return None if pd.isna(col) else col


def processing_csv(csv_content):
    if "Unnamed: 0" in csv_content:
        csv_content = csv_content.drop(columns=["Unnamed: 0"])
    return table_from_pandas(csv_content)


def _read_file(filename):
    """Reads both csv and json as structured json"""
    if str(filename).endswith(".csv"):
        csv_content = pd.read_csv(filename)
        return processing_csv(csv_content)
    return read_json(filename)


def read_file(filename):
    data = _read_file(filename)
    return fix_na(data)
