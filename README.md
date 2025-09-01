# graph_builder

Graph builder package.

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
