# ContextClue Graph Builder

ContextClue Graph Builder is an open-source toolkit for extracting knowledge graphs from semi-structured and unstructured data such as PDFs, reports, and tabular files.

It enables engineers, businesses, researchers, and developers to transform raw documents into graph structures for analytics, search, chatbots, and digital twin applications.

## Feaatures

📄 Document → Graph
Extract tabular information from documents and load it into graph structures.

⚙️ Flexible Configuration
Define headers, file paths, entity labels, and relationship types.

🚀 FastAPI Backend
Deploy graph extraction as a REST API service (Docker-ready).

🔄 Runtime Graph Retention
Graphs persist between API calls while the service is running.

🔮 Future Roadmap

* Automatic header extraction (semantic + layout)

* Smarter chunking & embeddings

* Integration with graph DBs and vector DBs

* Relationship discovery across multiple data sources

* Knowledge graph visualization dashboards

* RAG-enabled chatbot & business assistants

## Business Use Cases

💼 Business Use Cases

ContextClue Graph Builder goes beyond raw graph extraction—it powers enterprise-grade knowledge systems.

1. Industrial Engineering & Manufacturing

* Convert CAD, ERP, PLM, and planning data into unified, searchable knowledge graphs.
* Enable digital twin navigation: interactive exploration of components, processes, and relationships.
* Provide graph-based operational intelligence for predictive performance and system optimization.

2. Maintenance, Repair & Operations (MRO)

* Automotive, aerospace, energy, and logistics sectors use ContextClue to:
    * Reduce downtime with faster diagnostics.
    * Support predictive maintenance.
    * Increase efficiency of maintenance workflows.

3. Knowledge Assistants

* Integrate with chat platforms like Slack to build internal assistants.
* Example: Addeptalk (powered by ContextClue) connects Google Drive docs to Slack, enabling employees to ask natural-language questions and receive contextual answers.

4. Domain-Specific Applications

* Finance & Legal: Compliance document automation, audit preparation.
* Healthcare & Research: Extract structured knowledge from scientific papers and clinical reports.
* Developers & IT: Summarize technical docs, generate structured code knowledge, power RAG-based bots.


## Installation

Following good practices we suggest you create a separate virtual environment for working with Graph builder package.

Note that the graph_builder requires python 3.12 or higher.

```
poetry install
```

## Usage

Note that this is a short example taken from `examples/example1.ipynb`, for more information please
refer to it.

```python
from entity_graph.graph_extractor.entities_graph_extractor import EntitiesGraphExtractor

# Initialize the Extractor
extractor = EntitiesGraphExtractor()

test_data_path = "" # replace it with a correct data path

# Specify extraction configuration
config = {
    "extraction_type": "table_from_header",
    "filename": test_data_path + "coffee_machines.pdf",
    "header": ['Manufacturer', 'Coffee Machine Name', 'Machine ID', 'Production Year', 'Machine Type', 'Power (W)', 'Pressure (bar)', 'Water Tank Capacity (L)', 'Additional Features'],
}

# Load table to graph
extractor.load_table_from_file(
    config,
    "coffee_machines.pdf",
    "Machines",
    "instances",
)
```

## graph_builder FastApi

In the folder containing the `docker-compose.yml` file, run the commands:

```
docker compose build
```

Once the image is built:

```
docker compose up
```

Make sure to create the `.env` file in the directory based on the `.env_example` file with the needed environmental values.

## Important notes

In the current version of the application, graphs are retained between requests but not preserved across API restarts.

This means that each time the API is restarted, the graphs must be rebuilt.


## Roadmap

📌 Roadmap

 * Automatic header extraction (semantic segmentation + separators)

 * Improved data chunking and embeddings

 * Database and vector database infrastructure

 * Advanced relational analysis between sources

 * Interactive knowledge graph visualization


## Contributing

We welcome community contributions!

Fork this repo

Create a branch (feature/my-feature)

Commit changes (git commit -m "Add feature")

Push branch (git push origin feature/my-feature)

Open a Pull Request 🎉

Please include tests for new functionality.

 Integrated chatbot with RAG

