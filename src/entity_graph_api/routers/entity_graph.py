import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.entity_graph_api.models.entity_graph import LoadTableRequest, ExtractEntitiesRequest
from entity_graph.graph_extractor.entities_graph_extractor import EntitiesGraphExtractor

extractor = EntitiesGraphExtractor()
entities = extractor.entities_graph_manager.entities

UPLOAD_DIR = "/app/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(
    prefix="/entity_graph",
    tags=["entity_graph"],
    responses={404: {"description": "Not found"}},
)

@router.get("/test")
def test_entity_graph() -> dict[str, str]:
    """Test endpoint for the entity graph."""
    return {"message": "Entity graph is working!"}


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload any file and save it in the container."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"filename": file.filename, "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


@router.post("/load-table")
async def load_table(request: LoadTableRequest):
    """Load a table from an uploaded file into the graph."""
    try:
        file_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        extractor.load_table_from_file(request.config, file_path, request.table_name, request.instance_type)
        return {"message": "Table loaded successfully", "table_name": request.table_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load table: {e}")


@router.post("/extract-entities")
async def extract_entities(request: ExtractEntitiesRequest):
    """Extract entities and build the graph based on the provided steps."""
    try:
        extractor.extract_entities_graph(request.steps)
        return {"message": "Entities extracted and graph built successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract entities: {e}")


@router.get("/export-graph")
async def export_graph():
    """Export the graph data."""
    try:
        graph_export = extractor.export_graph()
        return graph_export
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export graph: {e}")


@router.get("/list-relations-for-entity")
async def list_relations_for_entity(entity_id: str):
    """List all relations for a specific entity."""
    try:
        relations = extractor.entities_graph_manager.find_entity(entity_id).relations
        return {"entity_id": entity_id, "relations": relations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list relations for entity: {e}")


@router.get("/list-named-entity-relations")
async def list_named_entity_relations(entity_name: str, relation_type: str):
    """List all named relations for a specific entity."""
    try:
        relations = extractor.entities_graph_manager.get_named_entity_relations(entity_name, relation_type)
        named_entity_relations = [e.name for e in relations]
        return {"entity_name": entity_name, "relation_type": relation_type, "named_relations": named_entity_relations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list named relations for entity: {e}")