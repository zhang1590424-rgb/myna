from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .compare import build_comparison
from .data_validation import DatasetValidationError
from .dataset_manager import DatasetManager
from .domain import (
    BatchVariableRequest,
    CreateExperimentRequest,
    DatasetFormat,
    LabBatchTestRequest,
    LabChatRequest,
    LabCompareChatRequest,
    LabLoadRequest,
    UpdateExperimentRequest,
)
from .downloader import ModelDownloader
from .engine import build_engine, real_engine_ready
from .environment import collect_environment_status
from .experiment_service import ExperimentService
from .inference_engine import InferenceEngine, resolve_gen_params
from .model_registry import get_model, get_model_catalog
from .paths import WEB_DIR, ensure_runtime_dirs
from .persistence import Database
from .queue_manager import QueueManager
from .templates import get_template, get_templates, get_training_presets, sample_csv_for_template


ensure_runtime_dirs()

app = FastAPI(title="个人训练工作台本地服务")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

db = Database()
datasets = DatasetManager(db)
experiments = ExperimentService(db, datasets)
training_engine = build_engine(experiments, datasets)
queue = QueueManager(experiments, training_engine)
lab = InferenceEngine(experiments)
model_downloader = ModelDownloader()


@app.on_event("startup")
async def _startup() -> None:
    queue.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await queue.shutdown()
    db.close()


# --------------------------------------------------------------------------- #
# Static + meta
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "assets" / "favicon.ico", media_type="image/x-icon")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/environment")
def environment():
    status = collect_environment_status()
    payload = status.model_dump()
    ready = real_engine_ready()
    payload["engine"] = "llamafactory" if ready else "not_ready"
    payload["engine_label"] = "真实训练" if ready else "环境未就绪，暂不能训练"
    payload["can_train"] = ready
    return payload


@app.get("/api/templates")
def templates():
    return get_templates()


@app.get("/api/training-presets")
def training_presets():
    return get_training_presets()


@app.get("/api/sample-data/{template_id}")
def sample_data(template_id: str) -> Response:
    try:
        template = get_template(template_id)
        csv_text = sample_csv_for_template(template_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个示例模板。") from exc
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{template.sample_filename}"'},
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@app.get("/api/models")
def models():
    return get_model_catalog()


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


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@app.get("/api/datasets")
def list_datasets():
    return datasets.list_datasets()


@app.post("/api/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    format: DatasetFormat = Form("alpaca"),
    name: str | None = Form(None),
):
    content = await file.read()
    filename = file.filename or "dataset"
    try:
        return datasets.save_upload(filename=filename, content=content, fmt=format, name=name)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=400, detail={"message": exc.message, "warnings": exc.warnings}) from exc


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        info = datasets.get_info(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个数据集。") from exc
    if info.format == "dpo_pairs":
        preview = [record.model_dump() for record in datasets.read_preferences(dataset_id)[:5]]
    else:
        preview = [record.model_dump() for record in datasets.read_records(dataset_id)[:5]]
    return {"info": info.model_dump(), "preview": preview}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    if not datasets.delete(dataset_id):
        raise HTTPException(status_code=404, detail="没有找到这个数据集。")
    return {"deleted": dataset_id}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
@app.get("/api/experiments")
def list_experiments():
    return experiments.list()


@app.post("/api/experiments")
async def create_experiment(request: CreateExperimentRequest):
    _validate_experiment_inputs(request.model_id, request.dataset_id, request.method)
    exp = experiments.create(request)
    queue.enqueue(exp.id)
    return experiments.get(exp.id)


@app.post("/api/experiments/batch")
async def batch_create_experiments(request: BatchVariableRequest):
    _validate_experiment_inputs(request.model_id, request.dataset_id, request.method)
    if not request.values:
        raise HTTPException(status_code=400, detail="请至少给变量一个取值。")
    created = experiments.batch_create(request)
    for exp in created:
        queue.enqueue(exp.id)
    return [experiments.get(exp.id) for exp in created]


@app.get("/api/experiments/compare")
def compare_experiments(ids: str):
    id_list = [i for i in ids.split(",") if i]
    try:
        return build_comparison(experiments, id_list)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有可对比的实验。") from exc


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str):
    try:
        return experiments.get(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc


@app.patch("/api/experiments/{exp_id}")
def update_experiment(exp_id: str, request: UpdateExperimentRequest):
    try:
        return experiments.update(exp_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc


@app.post("/api/experiments/{exp_id}/clone")
async def clone_experiment(exp_id: str):
    try:
        clone = experiments.clone(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    return clone


@app.post("/api/experiments/{exp_id}/stop")
async def stop_experiment(exp_id: str):
    try:
        return await training_engine.stop(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc


@app.delete("/api/experiments/{exp_id}")
def delete_experiment(exp_id: str):
    if not experiments.delete(exp_id):
        raise HTTPException(status_code=404, detail="没有找到这个实验。")
    return {"deleted": exp_id}


@app.get("/api/experiments/{exp_id}/export")
async def export_experiment(exp_id: str, merge: bool = False) -> FileResponse:
    try:
        result = await training_engine.export(exp_id, merge=merge)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(path=result.path, media_type=result.media_type, filename=result.filename)


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #
@app.get("/api/queue")
def queue_status():
    return queue.status()


@app.post("/api/queue/pause")
def pause_queue():
    queue.pause()
    return queue.status()


@app.post("/api/queue/resume")
def resume_queue():
    queue.resume()
    return queue.status()


# --------------------------------------------------------------------------- #
# Lab
# --------------------------------------------------------------------------- #
@app.get("/api/lab/status")
def lab_status():
    return lab.status()


@app.post("/api/lab/load")
def lab_load(request: LabLoadRequest):
    try:
        return lab.load(request.experiment_id, request.use_adapter)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/lab/unload")
def lab_unload():
    return lab.unload()


@app.post("/api/lab/chat")
async def lab_chat(request: LabChatRequest):
    try:
        answer = await lab.chat(request.prompt, request.max_new_tokens)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"prompt": request.prompt, "answer": answer}


@app.post("/api/lab/compare-chat")
async def lab_compare_chat(request: LabCompareChatRequest):
    gen_params = resolve_gen_params(
        request.style,
        {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty,
            "no_repeat_ngram_size": request.no_repeat_ngram_size,
        },
    )
    try:
        result = await lab.compare_chat(request.prompt, request.max_new_tokens, gen_params)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result


@app.post("/api/lab/batch-test")
async def lab_batch_test(request: LabBatchTestRequest):
    try:
        status = lab.status()
        if not status.loaded or not lab._experiment_id:
            raise RuntimeError("请先加载一个实验。")
        exp = experiments.get(lab._experiment_id)

        # 如果用户没传 prompts，从训练数据集里取前 5 条问题
        prompts = request.prompts
        if not prompts:
            try:
                records = datasets._read_rows(exp.dataset_id)
                prompts = [
                    r.get("instruction", r.get("query", ""))
                    for r in records[:5]
                    if r.get("instruction") or r.get("query")
                ]
            except Exception:
                pass
            if not prompts:
                prompts = ["你好，请介绍一下你自己。"]

        results = []
        for prompt in prompts:
            result = await lab.compare_chat(prompt, request.max_new_tokens)
            results.append(result)

        return {
            "experiment_id": exp.id,
            "experiment_name": exp.name,
            "results": results,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_experiment_inputs(model_id: str, dataset_id: str, method: str) -> None:
    if not real_engine_ready():
        raise HTTPException(
            status_code=409,
            detail="还不能开始训练。请先在模型页下载一个本地模型，并确认训练组件已就绪。",
        )

    try:
        model = get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个模型。") from exc
    if not model.available:
        raise HTTPException(status_code=409, detail="这个模型还没准备好，请先在模型页下载它。")

    try:
        info = datasets.get_info(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个数据集。") from exc

    expected = "dpo_pairs" if method == "dpo" else "alpaca"
    if info.format != expected:
        if method == "dpo":
            raise HTTPException(status_code=400, detail="DPO 需要偏好数据（chosen/rejected），这份是问答数据。")
        raise HTTPException(status_code=400, detail="SFT 需要问答数据，这份是偏好数据。")
    if info.row_count < 3:
        raise HTTPException(status_code=400, detail="有效数据太少了（至少 3 条），再补一些训练才稳。")
