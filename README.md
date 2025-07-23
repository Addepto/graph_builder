# graph_builder
Graph builder package. 

## Installation
```
poetry install
```

## Usage
```python
from entity_graph import EntitiesGraphExtractor

builder = EntitiesGraphExtractor()
builder.load_table_from_file(config)

builder.entities_graph_manager.export_objects_graph(collection="default")
```
