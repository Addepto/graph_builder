# graph_builder

Graph Builder

Graph Builder is an open-source toolkit for extracting structured knowledge graphs from documents and tabular data.
It enables you to transform raw data into graph structures for further analysis, visualization, and knowledge discovery.

## Feaatures

✨ Features

📄 Extract tables from documents and load them into a graph.

⚙️ Customizable extraction configurations (headers, file paths, entity names).

🔄 FastAPI integration for serving graph extraction as a service.

🗂️ Graphs are retained between requests during runtime.

🚀 Future roadmap includes:

    Automatic header extraction
    
    Smarter chunking and embeddings
    
    Database + vector database integration
    
    Advanced relationship discovery
    
    Knowledge graph visualization
    
    Chatbot + Retrieval-Augmented Generation (RAG)


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

 Improved data chunking and embeddings

 Database and vector database infrastructure

 Advanced relational analysis between sources

 Interactive knowledge graph visualization

 Integrated chatbot with RAG

