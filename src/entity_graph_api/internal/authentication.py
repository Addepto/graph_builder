from fastapi import HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from entity_graph_api.config import CONFIG, setup_logger

logger = setup_logger(__name__)

def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if not CONFIG.API_KEY:
        logger.warning("API_KEY is not set in the environment variable! sample is insecure!")
        return
    if not credentials:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if credentials.credentials != CONFIG.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials
