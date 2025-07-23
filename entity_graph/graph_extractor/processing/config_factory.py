from entity_graph.graph_extractor.processing.models import (
    ExtractionType,
    TableCsvContextExtracionConfig,
    TableFromHeaderExtracionConfig,
    TableMultilineExtracionConfig,
    TableWithContextExtracionConfig,
    TableXlsExtracionConfig,
    TextExtracionConfig,
    TxtExtracionConfig,
)

config_types_map = {
    ExtractionType.TABLE_FROM_HEADER: TableFromHeaderExtracionConfig,
    ExtractionType.TABLE_MULTILINE: TableMultilineExtracionConfig,
    ExtractionType.TABLE_WITH_CONTEXT: TableWithContextExtracionConfig,
    ExtractionType.XLS: TableXlsExtracionConfig,
    ExtractionType.CSV: TableCsvContextExtracionConfig,
    ExtractionType.TEXT: TextExtracionConfig,
    ExtractionType.TXT: TxtExtracionConfig,
}


def config_factory(config_dict):
    if config_dict.get("extraction_type") in config_types_map:
        config = config_types_map[config_dict.get("extraction_type")].model_validate(
            config_dict
        )
        # TODO: prints
        print(f"doing config: {config}")
        return config

    raise ValueError(f"extraction_type={config_dict.get('extraction_type')} unknown")
