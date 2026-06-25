from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

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
    LabResult,
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


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """对本地前端静态资源禁用 HTTP 缓存，确保 WKWebView 每次加载最新文件。"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(NoCacheStaticMiddleware)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

db = Database()
datasets = DatasetManager(db)
experiments = ExperimentService(db, datasets)
training_engine = build_engine(experiments, datasets)
queue = QueueManager(experiments, training_engine)
lab = InferenceEngine(experiments)
model_downloader = ModelDownloader()

# 孤儿训练自检巡检周期（秒）
ORPHAN_RECONCILE_INTERVAL = 30
_orphan_task: asyncio.Task[None] | None = None


async def _orphan_reconcile_loop() -> None:
    """周期回收异常退出的训练实验，避免状态卡在 running 让用户无法删除。"""
    while True:
        try:
            await asyncio.sleep(ORPHAN_RECONCILE_INTERVAL)
            training_engine.reconcile_orphans()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 自检失败不影响主服务
            continue


@app.on_event("startup")
async def _startup() -> None:
    # 服务启动时先回收一次：上次进程异常退出留下的「孤儿训练」会被扳成 failed
    training_engine.reconcile_orphans()
    queue.start()
    global _orphan_task
    _orphan_task = asyncio.create_task(_orphan_reconcile_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _orphan_task
    if _orphan_task is not None:
        _orphan_task.cancel()
        try:
            await _orphan_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        _orphan_task = None
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
    ready = real_engine_ready(status)
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
        preview = [record.model_dump() for record in datasets.read_preferences(dataset_id)[:20]]
    else:
        preview = [record.model_dump() for record in datasets.read_records(dataset_id)[:20]]
    return {"info": info.model_dump(), "preview": preview}


@app.put("/api/datasets/{dataset_id}")
async def update_dataset(dataset_id: str, file: UploadFile = File(...)):
    try:
        info = datasets.get_info(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个数据集。") from exc
    content = await file.read()
    filename = file.filename or info.source_filename
    try:
        result = datasets.update_dataset(dataset_id, filename=filename, content=content, fmt=info.format)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=400, detail={"message": exc.message, "warnings": exc.warnings}) from exc
    return result


@app.get("/api/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str):
    try:
        info = datasets.get_info(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个数据集。") from exc
    path = datasets.records_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="数据文件不存在。")
    return FileResponse(path, filename=f"{info.name}.json", media_type="application/json")


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    if not datasets.delete(dataset_id):
        raise HTTPException(status_code=404, detail="没有找到这个数据集。")
    return {"deleted": dataset_id}


# --------------------------------------------------------------------------- #
# Pre-training diagnostics
# --------------------------------------------------------------------------- #
@app.get("/api/datasets/{dataset_id}/preflight")
def preflight_check(dataset_id: str, method: str = "sft"):
    from .diagnostics import preflight_data_check

    try:
        rows = datasets._read_rows(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个数据集。") from exc
    cards = preflight_data_check(rows, method=method)
    return {"cards": [c.model_dump() for c in cards]}


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
        exp = experiments.get(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    payload = exp.model_dump()
    # Attach live diagnostics for running experiments
    if exp.status in ("running",) and exp.loss:
        from .diagnostics import compute_live_diagnostics

        live = compute_live_diagnostics(exp)
        payload["live_diagnostics"] = [c.model_dump() for c in live]
    else:
        payload["live_diagnostics"] = []
    return payload


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
    try:
        deleted = experiments.delete(exp_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
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


@app.get("/api/lab/history")
def lab_history(experiment_id: str | None = None):
    target_id = experiment_id or lab.status().experiment_id
    if not target_id:
        return {"experiment_id": None, "results": []}
    return {"experiment_id": target_id, "results": db.list_lab_results(target_id)}


@app.get("/api/lab/history/batch")
def lab_history_batch(experiment_ids: str = "", limit: int = 100):
    """一次查询多个实验的测评历史，避免前端 N+1 请求。"""
    ids = [i.strip() for i in experiment_ids.split(",") if i.strip()]
    if not ids:
        return {"results": []}
    safe_limit = max(1, min(limit, 200))
    return {"results": db.list_lab_results_batch(ids, safe_limit)}


@app.get("/api/lab/history/recent")
def lab_history_recent(limit: int = 6):
    safe_limit = max(1, min(limit, 20))
    return {"results": db.list_recent_lab_results(safe_limit)}


@app.delete("/api/lab/history/{result_id}")
def lab_history_delete(result_id: str):
    if not db.delete_lab_result(result_id):
        raise HTTPException(status_code=404, detail="没有找到这条测评记录。")
    return {"ok": True}


@app.get("/api/lab/history/{result_id}")
def lab_history_get(result_id: str):
    result = db.get_lab_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="没有找到这条测评记录。")
    return result


@app.get("/api/lab/starters")
def lab_starters(experiment_id: str | None = None):
    target_id = experiment_id or lab.status().experiment_id
    if not target_id:
        return {"experiment_id": None, "prompts": []}
    try:
        exp = experiments.get(target_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    return {"experiment_id": exp.id, "prompts": _starter_prompts_for_experiment(exp.dataset_id)}


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
    status = lab.status()
    if status.experiment_id:
        _save_lab_result(
            experiment_id=status.experiment_id,
            experiment_name=status.experiment_name or "未命名实验",
            kind="compare",
            result=result,
            extra={"style": request.style, "gen_params": gen_params, "metrics": result.get("metrics", {})},
        )
    return result


@app.post("/api/lab/compare-runs")
async def lab_compare_run(request: LabCompareChatRequest):
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
        status = lab.status()
        if not status.loaded or not lab._experiment_id:
            raise RuntimeError("请先加载一个实验。")
        exp = experiments.get(lab._experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    pending = _save_lab_result(
        experiment_id=exp.id,
        experiment_name=exp.name,
        kind="compare",
        result={"prompt": request.prompt, "base_answer": "", "finetuned_answer": ""},
        extra={"status": "running", "style": request.style, "gen_params": gen_params},
    )
    asyncio.create_task(
        _finish_compare_lab_result(pending.id, exp.id, request.prompt, request.max_new_tokens, gen_params)
    )
    return pending


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

        source = "manual" if request.prompts else "dataset-sample"
        _save_lab_result(
            experiment_id=exp.id,
            experiment_name=exp.name,
            kind="batch",
            result={
                "prompt": f"批量测评（{len(results)} 条）",
                "base_answer": "",
                "finetuned_answer": "",
            },
            extra={"source": source, "results": results},
        )

        return {
            "experiment_id": exp.id,
            "experiment_name": exp.name,
            "results": results,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/lab/batch-runs")
async def lab_batch_run(request: LabBatchTestRequest):
    try:
        status = lab.status()
        if not status.loaded or not lab._experiment_id:
            raise RuntimeError("请先加载一个实验。")
        exp = experiments.get(lab._experiment_id)
        prompts = _lab_batch_prompts(exp.dataset_id, request.prompts)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    source = "manual" if request.prompts else "dataset-sample"
    pending = _save_lab_result(
        experiment_id=exp.id,
        experiment_name=exp.name,
        kind="batch",
        result={"prompt": f"批量测评（{len(prompts)} 条）", "base_answer": "", "finetuned_answer": ""},
        extra={"status": "running", "source": source, "results": [], "prompts": prompts},
    )
    asyncio.create_task(_finish_batch_lab_result(pending.id, exp.id, prompts, request.max_new_tokens, source))
    return pending


# --------------------------------------------------------------------------- #
# Session-based chat (persistent model process)
# --------------------------------------------------------------------------- #
@app.post("/api/lab/session/start")
async def lab_session_start(request: LabLoadRequest):
    """启动常驻推理进程，加载模型到内存。"""
    gen_params = resolve_gen_params("balanced")
    try:
        result = await lab.start_session(request.experiment_id, gen_params)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="没有找到这个实验。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result


@app.post("/api/lab/session/chat")
async def lab_session_chat(request: dict):
    """向常驻进程发送对话请求。body: {messages, max_new_tokens}。"""
    messages = request.get("messages", [])
    max_new_tokens = request.get("max_new_tokens", 120)
    if not messages:
        raise HTTPException(status_code=400, detail="messages 不能为空。")
    try:
        result = await lab.session_chat(messages, max_new_tokens)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result


@app.post("/api/lab/session/end")
async def lab_session_end(request: dict | None = None):
    """结束对话 session，释放模型内存。可选保存对话记录。"""
    body = request or {}
    messages = body.get("messages", [])
    experiment_id = body.get("experiment_id")

    # 保存对话记录
    if messages and experiment_id:
        try:
            exp = experiments.get(experiment_id)
            first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "自由对话")
            _save_lab_result(
                experiment_id=exp.id,
                experiment_name=exp.name,
                kind="chat",
                result={"prompt": first_user_msg[:80], "base_answer": "", "finetuned_answer": ""},
                extra={"messages": messages, "status": "completed"},
            )
        except (KeyError, StopIteration):
            pass

    await lab.end_session()
    return {"ok": True}


@app.get("/api/lab/session/status")
def lab_session_status():
    """查询当前 session 状态。"""
    return {
        "active": lab.session_active,
        "experiment_id": lab._experiment_id if lab.session_active else None,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_lab_result(
    experiment_id: str,
    experiment_name: str,
    kind: Literal["compare", "batch", "chat"],
    result: dict,
    extra: dict | None = None,
) -> LabResult:
    lab_result = LabResult(
        id=uuid.uuid4().hex,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        kind=kind,
        prompt=result.get("prompt", ""),
        base_answer=result.get("base_answer", ""),
        finetuned_answer=result.get("finetuned_answer", ""),
        created_at=datetime.now().isoformat(timespec="seconds"),
        data=extra or {},
    )
    db.upsert_lab_result(lab_result)
    return lab_result


def _update_lab_result(result_id: str, result: dict, extra: dict) -> None:
    saved = db.get_lab_result(result_id)
    if saved is None:
        return
    result_extra = {}
    if result.get("metrics"):
        result_extra["metrics"] = result["metrics"]
    db.upsert_lab_result(
        saved.model_copy(
            update={
                "prompt": result.get("prompt", saved.prompt),
                "base_answer": result.get("base_answer", saved.base_answer),
                "finetuned_answer": result.get("finetuned_answer", saved.finetuned_answer),
                "data": {**saved.data, **result_extra, **extra},
            }
        )
    )


async def _finish_compare_lab_result(
    result_id: str,
    experiment_id: str,
    prompt: str,
    max_new_tokens: int,
    gen_params: dict,
) -> None:
    try:
        lab.load(experiment_id, True)
        result = await lab.compare_chat(prompt, max_new_tokens, gen_params)
        _update_lab_result(result_id, result, {"status": "completed"})
    except Exception as exc:
        _update_lab_result(
            result_id,
            {"prompt": prompt, "base_answer": "", "finetuned_answer": ""},
            {"status": "failed", "error": str(exc)},
        )


async def _finish_batch_lab_result(
    result_id: str,
    experiment_id: str,
    prompts: list[str],
    max_new_tokens: int,
    source: str,
) -> None:
    results = []
    try:
        lab.load(experiment_id, True)
        for prompt in prompts:
            results.append(await lab.compare_chat(prompt, max_new_tokens))
        _update_lab_result(
            result_id,
            {"prompt": f"批量测评（{len(results)} 条）", "base_answer": "", "finetuned_answer": ""},
            {"status": "completed", "source": source, "results": results},
        )
    except Exception as exc:
        _update_lab_result(
            result_id,
            {"prompt": f"批量测评（{len(prompts)} 条）", "base_answer": "", "finetuned_answer": ""},
            {"status": "failed", "source": source, "results": results, "error": str(exc)},
        )


def _lab_batch_prompts(dataset_id: str, request_prompts: list[str]) -> list[str]:
    prompts = [p.strip() for p in request_prompts if p.strip()]
    if prompts:
        return prompts
    try:
        records = datasets._read_rows(dataset_id)
        prompts = [
            r.get("instruction", r.get("query", ""))
            for r in records[:5]
            if r.get("instruction") or r.get("query")
        ]
    except Exception:
        prompts = []
    return prompts or ["你好，请介绍一下你自己。"]


def _starter_prompts_for_experiment(dataset_id: str, limit: int = 4) -> list[str]:
    fallback = ["你好，请介绍一下你自己。"]
    try:
        rows = datasets._read_rows(dataset_id)
    except Exception:
        return fallback

    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        raw = row.get("instruction") or row.get("query") or row.get("question") or ""
        prompt = str(raw).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt[:120])

    if not prompts:
        return fallback
    if len(prompts) <= limit:
        return prompts

    # Deterministic spread across the dataset instead of always showing the first rows.
    step = (len(prompts) - 1) / (limit - 1)
    return [prompts[round(i * step)] for i in range(limit)]


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
