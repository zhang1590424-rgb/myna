from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .data_validation import DatasetValidationError
from .domain import CompareRequest, CreateTrainingJobRequest
from .downloader import ModelDownloader
from .engine import JobStore, build_engine
from .environment import collect_environment_status
from .llamafactory import LlamaFactoryConfigBuilder
from .paths import WEB_DIR, ensure_runtime_dirs
from .store import DatasetStore
from .templates import (
    get_model,
    get_model_catalog,
    get_template,
    get_templates,
    get_training_presets,
    sample_csv_for_template,
)


ensure_runtime_dirs()

app = FastAPI(title="小白训练师本地服务")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

dataset_store = DatasetStore()
job_store = JobStore()
training_engine = build_engine(job_store, dataset_store)
config_builder = LlamaFactoryConfigBuilder()
model_downloader = ModelDownloader()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/environment")
def environment():
    status = collect_environment_status()
    payload = status.model_dump()
    payload["engine"] = training_engine.name
    payload["engine_label"] = "真实训练" if training_engine.name == "llamafactory" else "演示模式"
    return payload


@app.get("/api/templates")
def templates():
    return get_templates()


@app.get("/api/models")
def models():
    return get_model_catalog()


@app.get("/api/training-presets")
def training_presets():
    return get_training_presets()


@app.post("/api/models/{model_id}/download")
async def download_model(model_id: str):
    try:
        model = get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个模型。") from exc
    if model.available:
        raise HTTPException(status_code=409, detail="这个模型已经在本地了，可以直接使用。")
    if not model.repo_id:
        raise HTTPException(status_code=409, detail="这个模型暂不支持自动下载。")
    return await model_downloader.start(model_id, model.repo_id)


@app.get("/api/models/{model_id}/download")
def download_status(model_id: str):
    try:
        get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个模型。") from exc
    return model_downloader.status_for(model_id)


@app.get("/api/sample-data/{template_id}")
def sample_data(template_id: str) -> Response:
    try:
        template = get_template(template_id)
        csv_text = sample_csv_for_template(template_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个模板。") from exc

    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{template.sample_filename}"'},
    )


@app.post("/api/datasets/validate")
async def validate_dataset(
    template_id: str = Form(...),
    file: UploadFile = File(...),
):
    content = await file.read()
    filename = file.filename or "dataset"
    try:
        get_template(template_id)
        return dataset_store.save_upload(filename=filename, content=content, template_id=template_id)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=400, detail={"message": exc.message, "warnings": exc.warnings}) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个模板。") from exc


@app.post("/api/training/jobs")
async def create_training_job(request: CreateTrainingJobRequest):
    try:
        get_template(request.template_id)
        model = get_model(request.model_id)
        metadata = dataset_store.read_metadata(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="模板、模型或数据不存在。") from exc

    if not model.available:
        raise HTTPException(status_code=409, detail="这个模型还没准备好，请先在准备环境页下载它。")

    valid_count = int(metadata["valid_count"])
    if valid_count < 3:
        raise HTTPException(
            status_code=400,
            detail="有效数据太少了（至少要 3 条）。再补一些问答数据，训练效果才稳。",
        )

    job = await job_store.create(
        template_id=request.template_id,
        dataset_id=request.dataset_id,
        model_id=request.model_id,
        dataset_count=valid_count,
        settings=request.settings,
        engine=training_engine.name,
    )
    await training_engine.start(job)
    return job


@app.get("/api/training/jobs")
async def list_training_jobs():
    return await job_store.list()


@app.get("/api/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    try:
        return await job_store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个训练任务。") from exc


@app.post("/api/training/jobs/{job_id}/stop")
async def stop_training_job(job_id: str):
    try:
        return await training_engine.stop(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个训练任务。") from exc


@app.post("/api/compare")
async def compare(request: CompareRequest):
    try:
        return await training_engine.compare(request.job_id, request.prompt)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个训练任务。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/training/jobs/{job_id}/export")
async def export_training_job(job_id: str, merge: bool = False) -> FileResponse:
    try:
        result = await training_engine.export(job_id, merge=merge)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个训练任务。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return FileResponse(
        path=result.path,
        media_type=result.media_type,
        filename=result.filename,
    )


@app.post("/api/training/jobs/{job_id}/llamafactory-preview")
async def preview_llamafactory_config(job_id: str):
    try:
        job = await job_store.get(job_id)
        records = dataset_store.read_records(job.dataset_id)
        template = get_template(job.template_id)
        model = get_model(job.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="训练任务、数据、模板或模型不存在。") from exc

    prepared = config_builder.prepare(job=job, records=records, template=template, model=model)
    return {
        "config_file": str(prepared.config_file),
        "dataset_file": str(prepared.dataset_file),
        "dataset_info_file": str(prepared.dataset_info_file),
        "command": prepared.command,
    }
