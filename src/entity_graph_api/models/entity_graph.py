import os
from typing import List, Dict, Any
from pydantic import BaseModel, field_validator
from src.entity_graph_api.config import UPLOAD_DIR

class LoadTableRequest(BaseModel):
    config: Dict[str, Any] | str
    filename: str
    table_name: str
    instance_type: str

    @field_validator('config')
    @classmethod
    def config_must_have_filename(cls, v):
        if isinstance(v, dict):
            filename = v.get("filename")
            if not filename:
                raise ValueError('config dictionary must contain a "filename" field')
            full_path = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(full_path):
                raise ValueError(f'File not found: {full_path}')
            v["filename"] = full_path
            return v
        if isinstance(v, str):
            full_path = os.path.join(UPLOAD_DIR, v)
            if not os.path.exists(full_path):
                raise ValueError(f'File not found: {full_path}')
            return full_path
        return v

class LoadTableRequestWithString(BaseModel):
    config: str
    filename: str
    table_name: str
    instance_type: str

class ExtractEntitiesRequest(BaseModel):
    steps: Dict[str, List[Any]]