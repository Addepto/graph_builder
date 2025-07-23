from enum import StrEnum

from pydantic import BaseModel


class OutputDataType(StrEnum):
    JSON = "json"
    CSV = "csv"
    DATAFRAME = "dataframe"


class ExtractionType(StrEnum):
    TABLE_FROM_HEADER = "table_from_header"
    TABLE_MULTILINE = "table_multiline"
    TABLE_WITH_CONTEXT = "table_with_context"
    XLS = "xls"
    CSV = "csv"
    TEXT = "text"
    TXT = "txt"


class BaseExtractionConfig(BaseModel):
    data_type: OutputDataType
    filename: str
    extraction_type: ExtractionType


class TableFromHeaderExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.TABLE_FROM_HEADER
    header: list[str]
    pages: None | tuple[int, int] = None
    data_type: OutputDataType = OutputDataType.DATAFRAME
    columns_to_split: list | None = None
    prefixes: list | None = None


class TableMultilineExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.TABLE_MULTILINE
    pages: None | tuple[int, int] = None
    data_type: OutputDataType = OutputDataType.JSON
    prefixes: list | None = None


class TableWithContextExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.TABLE_WITH_CONTEXT
    pages: tuple[int, int]
    data_type: OutputDataType = OutputDataType.DATAFRAME
    context_label: str
    header_double: list[str | None] | None
    header_single: list[str | None] | None
    header_output: list[str] | None
    prefixes: list | None = None


class TableXlsExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.XLS
    labels_row_id: int = 0
    data_start_row_id: int = 1
    required_field: str | None = None
    prefixes: list | None = None
    normalize: list | None = None
    protected: bool = False
    data_type: OutputDataType = OutputDataType.DATAFRAME


class TableCsvContextExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.CSV
    data_type: OutputDataType = OutputDataType.DATAFRAME


class TextExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.TEXT
    data_type: OutputDataType = OutputDataType.JSON
    context_label: str
    pages: tuple[int, int]
    prefix: str | None = None


class TxtExtracionConfig(BaseExtractionConfig):
    extraction_type: ExtractionType = ExtractionType.TXT
    data_type: OutputDataType = OutputDataType.JSON
