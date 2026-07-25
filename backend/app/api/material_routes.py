from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ..materials import MaterialExtractionError, extract_materials
from ..schemas import MaterialExtractionResponse

router = APIRouter(prefix="/materials", tags=["materials"])

@router.post("/extract", response_model=MaterialExtractionResponse)
async def material_extract(files: list[UploadFile] = File(...)) -> MaterialExtractionResponse:
    payload: list[tuple[str, str, bytes]] = []
    for upload in files:
        payload.append((upload.filename or "unnamed", upload.content_type or "", await upload.read()))
    try:
        return await run_in_threadpool(extract_materials, payload)
    except MaterialExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
