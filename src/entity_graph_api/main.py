from fastapi import Depends, FastAPI

from entity_graph_api.internal.authentication import verify_api_key
from entity_graph_api.routers import health, entity_graph


app = FastAPI(debug=True)

app.include_router(health.router, dependencies=[Depends(verify_api_key)])
app.include_router(entity_graph.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Auth Boilerplate!"}
