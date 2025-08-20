from entity_graph.graph_extractor.processing.config_factory import config_factory
from entity_graph.graph_extractor.processing.enrichment import (
    add_enrichment_links,
    enrich_from_data_with_map,
    enrich_from_table_name,
    enrich_identifier_values,
)
from entity_graph.graph_extractor.processing.extraction import (
    create_instances,
    link_id,
    new_id,
)
from entity_graph.graph_extractor.processing.extraction_router import extraction
from entity_graph.graph_extractor.processing.ingesting import add_table_from_json
from entity_graph.graph_extractor.processing.model_structure import add_instances_info
from entity_graph.graph_extractor.processing.models import (
    BaseExtractionConfig,
    OutputDataType,
)
from entity_graph.graph_extractor.processing.utils import (
    fix_na,
    processing_csv,
    read_file,
)
from entity_graph.graph_manager import EntityManager
from entity_graph.models.entity_graph.file import File


class EntitiesGraphExtractor:

    def __init__(
        self, *args, representation=None, nodes=None, collection="default", **kwargs
    ):
        self.entities_graph_manager = EntityManager()
        self.documents_data = {}
        self.extraction_plan = None
        self.stages_done = set()
        self.file_sources = {}
        self.local = False
        self.collection = collection

    def load_table_from_file(
        self, source: str | dict, filename: str, table_name: str, data_type: str
    ):
        self._add_file_node(filename)
        self._add_table(self._read_source(source), filename, table_name, data_type)
        return

    def _add_file_node(self, filename):
        if self.entities_graph_manager.get_entity(filename):
            return
        file = File(
            None, filename, "instances", filename.split(".")[-1], "instances", -1
        )
        self.entities_graph_manager.add_entity(file, collection=self.collection)

    def _read_source(self, f: str | dict):
        """
        Supports filenames (json/csv) and `dict`s (extraction configs).
        Filenames are used to get data from `self.documents_data`.
        `dict`s are converted into extraction configs and used for extraction.
        If self.local, files from config are used directly. If not, loaded with `temp_file_manager`.
        All files need to be previously added/linked unless local extraction.
        """

        if isinstance(f, str):
            return read_file(f)

        print(f"Extraction! From {f}")

        config: BaseExtractionConfig = config_factory(f)

        content = extraction(config)

        if config.data_type in (OutputDataType.CSV, OutputDataType.DATAFRAME):
            content = processing_csv(content)
        return fix_na(content)

    def _add_table(self, json_data, file_name, table_name, data_type):
        # Pass the collection parameter to add_table_from_json
        table = add_table_from_json(
            self.entities_graph_manager,
            json_data,
            file_name,
            table_name,
            data_type=data_type,
            collection=self.collection,
        )
        # TODO: workaround
        if "model_" in table_name:
            model = table_name.split("model_")[1]
            # actions.append(("assign", table_name, "header", {"model": model}))
            setattr(table, "header", {"model": model})
            if hasattr(table, "collection"):
                table.collection = self.collection

    ##   OLD ONES

    def _add_identifiers(self):
        # Create new identifiers
        for name, table_name, cols, fill_values in self.extraction_plan.get(
            "identifiers", []
        ):
            new_id(
                self.entities_graph_manager,
                name,
                table_name,
                cols,
                fill_values,
                collection=self.collection,
            )

        # Link identifiers
        for name, table_name, cols, fill_values in self.extraction_plan.get(
            "identifiers_links", []
        ):
            link_id(
                self.entities_graph_manager,
                name,
                table_name,
                cols,
                fill_values,
                collection=self.collection,
            )

    def _make_instances(self):
        for (
            id_name,
            table_name,
            do_hierarchy,
            override_cols,
        ) in self.extraction_plan.get("instances_creation", []):
            # Make sure to pass the collection parameter
            create_instances(
                self.entities_graph_manager,
                id_name,
                table_name,
                do_hierarchy,
                override_cols=override_cols,
                collection=self.collection,  # Pass the collection parameter
            )

    def _enrichment_matching(self):
        """
        Perform enrichment matching based on the extraction plan.
        Uses self.collection for collection parameter.
        """
        for id_name, attr_name, key in self.extraction_plan.get("enrichment_links", []):
            # Pass the collection parameter
            add_enrichment_links(
                self.entities_graph_manager,
                id_name,
                attr_name,
                key,
                collection=self.collection,
            )

        for table_name, label1, label2, new_labels in self.extraction_plan.get(
            "enrichments", []
        ):
            # Pass the collection parameter
            enrich_from_table_name(
                self.entities_graph_manager,
                table_name,
                label1,
                label2,
                new_labels,
                collection=self.collection,
            )

    def _enrichment_models(self):
        """
        Perform enrichment with models based on the extraction plan.
        Uses self.collection for collection parameter.
        """
        for id_name in self.extraction_plan.get("identifiers_to_enrich", []):
            # Pass the collection parameter
            enrich_identifier_values(
                self.entities_graph_manager, id_name, collection=self.collection
            )

    def _enrichment_fuzzy(self):
        """
        Perform fuzzy enrichment based on the extraction plan.
        Uses self.collection for collection parameter.
        """
        # Refresh name map for the collection
        self.entities_graph_manager.refresh_name_map(collection=self.collection)

        for id_name, table_col, source_name, label, _map in self.extraction_plan.get(
            "enrichments_from_data", []
        ):
            # Find source entity in the specific collection
            source_entity = self.entities_graph_manager.named_entity(
                source_name, collection=self.collection
            )
            if source_entity is None:
                print(
                    f"Warning: Source entity '{source_name}' not found in collection '{self.collection}', skipping enrichment"
                )
                continue

            # Get table contents for this collection
            enrichment_data = self.entities_graph_manager.get_table_contents_dict(
                source_entity, label, collection=self.collection
            )

            # Find identifier in the specific collection
            _id = self.entities_graph_manager.named_entity(
                id_name, collection=self.collection
            )
            if _id is None:
                print(
                    f"Warning: Identifier '{id_name}' not found in collection '{self.collection}'"
                )
                continue

            # Process values
            for name in list(_id.data["values"]):
                # Find object in the specific collection
                obj = self.entities_graph_manager.named_entity(
                    name, collection=self.collection
                )
                if obj is not None:
                    # Pass the collection parameter
                    enrich_from_data_with_map(
                        self.entities_graph_manager,
                        obj,
                        table_col,
                        _map,
                        enrichment_data,
                        collection=self.collection,
                    )
                else:
                    print(
                        f"Warning: Entity with name '{name}' not found in collection '{self.collection}'"
                    )

    def _join_identifiers(self):
        """
        For given attribute names tries to find matching entity names.
        If there is a match we add parent-child relationship.
        """
        self.entities_graph_manager.refresh_name_map(collection=self.collection)

        print(
            f"Name map has {len(self.entities_graph_manager.get_name_map(collection=self.collection))} entries in collection '{self.collection}'"
        )
        join_count = 0
        for linking_column in self.extraction_plan.get("linking_columns", []):
            print(
                f"Processing linking column: {linking_column} in collection '{self.collection}'"
            )

            # Filter entities by collection
            entities = [
                e
                for e_id, e in self.entities_graph_manager.entities.items()
                if getattr(
                    e, "collection", self.entities_graph_manager.default_collection
                )
                == self.collection
            ]

            for obj in entities:
                attrs = getattr(obj, "attributes", {})
                if not attrs or linking_column not in attrs:
                    continue

                station_number = attrs.get(linking_column)
                if not station_number:
                    continue

                station_number = str(station_number).strip()

                parent = self.entities_graph_manager.named_entity(
                    station_number, collection=self.collection
                )

                if not parent:
                    print(f"Parent not found, try find '{station_number}'")
                    parent = self.entities_graph_manager.find_entity(
                        station_number, collection=self.collection
                    )
                    if not parent:
                        print(f"Find failed as well for '{station_number}'")

                if parent and parent != obj:
                    parent.add_relation(obj, "parent_of")
                    obj.add_relation(parent, "child_of")
                    join_count += 1

                    print(
                        f"Joined {obj.name} as child of {parent.name} in collection '{self.collection}'"
                    )
                else:
                    print(
                        f"No parent found for {obj.name} with {linking_column}={station_number} in collection '{self.collection}'"
                    )

        print(
            f"Created {join_count} parent-child relationships in collection '{self.collection}'"
        )

    def _instances_context(self):
        for link_config in self.extraction_plan.get("instances_context", []):
            info_source_filename = link_config[2]
            info_source = self._read_file(info_source_filename)

            add_instances_info(
                self.entities_graph_manager,
                link_config,
                info_source,
                collection=self.collection,
            )

    def extract_entities_graph_by_stage(self, stages, extraction_plan_config=None):
        self._make_extraction_plan(extraction_plan_config)
        # Define a dictionary mapping stage names to their corresponding methods
        stages_methods = {
            "_add_identifiers": self._add_identifiers,
            "_make_instances": self._make_instances,
            "_enrichment_matching": self._enrichment_matching,
            "_enrichment_models": self._enrichment_models,
            "_enrichment_fuzzy": self._enrichment_fuzzy,
            "_join_identifiers": self._join_identifiers,
        }

        # Iterate over stages and execute corresponding methods
        # TODO: ensure order break after first missing after first done
        # define dependencies
        # refuse if dependencies not done
        for stage in stages_methods:
            if stage in stages:
                if stage not in self.stages_done:
                    stages_methods[stage]()
                self.stages_done.add(stage)

        return self

    # Update the extract_entities_graph2 method to check for completed steps
    def extract_entities_graph(
        self, extraction_plan_config: dict[str, list[tuple]] | None = None
    ):
        """
        Extract entities graph - second phase.

        :param extraction_plan_config: Optional configuration for extraction plan.
        :return: Self for method chaining.
        """
        self.extraction_plan = extraction_plan_config

        if "_add_identifiers" not in self.stages_done:
            self._add_identifiers()
            self.stages_done.add("_add_identifiers")

        if "_make_instances" not in self.stages_done:
            self._make_instances()
            self.stages_done.add("_make_instances")

        if "_enrichment_matching" not in self.stages_done:
            self._enrichment_matching()
            self.stages_done.add("_enrichment_matching")

        if "_enrichment_models" not in self.stages_done:
            self._enrichment_models()
            self.stages_done.add("_enrichment_models")

        if "_enrichment_fuzzy" not in self.stages_done:
            self._enrichment_fuzzy()
            self.stages_done.add("_enrichment_fuzzy")

        if "_join_identifiers" not in self.stages_done:
            self._join_identifiers()
            self.stages_done.add("_join_identifiers")

        if "_instances_context" not in self.stages_done:
            self._instances_context()
            self.stages_done.add("_instances_context")

        return self

    def export_graph(self) -> tuple[list[str], list[tuple[str, str]], dict[str, dict]]:
        """
        Export objects sub-graph to json.

        :return: Tuple of nodes, edges, and metadata
        """
        return self.entities_graph_manager.export_objects_graph(collection="default")
