from typing import List, Dict, Any
from pydantic import BaseModel

class LoadTableRequest(BaseModel):
    config: Dict[str, Any] | str
    filename: str
    table_name: str
    instance_type: str

class ExtractEntitiesRequest(BaseModel):
    steps: Dict[str, List[Any]]