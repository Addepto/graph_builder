from entity_graph.graph_extractor.processing.extraction_page import (
    get_pages_elements,
    read_txt,
)
from entity_graph.graph_extractor.processing.extraction_table import (
    get_all_matching_tables,
    get_excel_table,
    get_multiline_tables,
    get_tables_with_context,
)
from entity_graph.graph_extractor.processing.models import ExtractionType

extractions_map = {
    ExtractionType.TABLE_FROM_HEADER: get_all_matching_tables,
    ExtractionType.TABLE_MULTILINE: get_multiline_tables,
    ExtractionType.TABLE_WITH_CONTEXT: get_tables_with_context,
    ExtractionType.XLS: get_excel_table,
    ExtractionType.CSV: None,
    ExtractionType.TEXT: get_pages_elements,
    ExtractionType.TXT: read_txt,
}


def extraction(config):
    if config.extraction_type in extractions_map:
        return extractions_map[config.extraction_type](config)
    raise ValueError
