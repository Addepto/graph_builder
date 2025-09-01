from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["health_checks"],
    responses={404: {"description": "Not found"}},
)


@router.get("/live")
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify if the service is alive."""
    return {"status": "alive"}
