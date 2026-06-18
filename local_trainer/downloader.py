from __future__ import annotations

import asyncio
import shutil

from .domain import ModelDownloadStatus
from .paths import MODELS_DIR, model_dir_for_repo


class ModelDownloader:
    """Downloads models from ModelScope (domestic mirror) into MODELS_DIR.

    Progress is coarse: ModelScope's snapshot_download doesn't give a clean
    callback, so we report a single in-flight 'downloading' state and flip to
    'completed' once the weight files land. The work runs in a thread so the
    event loop is never blocked.
    """

    def __init__(self) -> None:
        self._status: dict[str, ModelDownloadStatus] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def status_for(self, model_id: str) -> ModelDownloadStatus:
        return self._status.get(model_id, ModelDownloadStatus(model_id=model_id))

    async def start(self, model_id: str, repo_id: str) -> ModelDownloadStatus:
        async with self._lock:
            current = self._status.get(model_id)
            if current and current.state == "downloading":
                return current
            status = ModelDownloadStatus(
                model_id=model_id,
                state="downloading",
                progress=3,
                message=f"正在从国内源下载 {repo_id}，请保持网络畅通。",
            )
            self._status[model_id] = status
            self._tasks[model_id] = asyncio.create_task(self._run(model_id, repo_id))
            return status

    async def _run(self, model_id: str, repo_id: str) -> None:
        try:
            free_gb = shutil.disk_usage(MODELS_DIR.parent).free / (1024**3)
            if free_gb < 2:
                self._status[model_id] = ModelDownloadStatus(
                    model_id=model_id,
                    state="failed",
                    message="磁盘空间不足，建议清理后再下载（至少留出几 GB）。",
                    error=f"only {free_gb:.1f} GB free",
                )
                return

            await asyncio.to_thread(self._download_blocking, repo_id)

            target = model_dir_for_repo(repo_id)
            ok = (target / "config.json").exists() and any(target.glob("*.safetensors"))
            if ok:
                self._status[model_id] = ModelDownloadStatus(
                    model_id=model_id,
                    state="completed",
                    progress=100,
                    message="下载完成，可以选择这个模型开始训练了。",
                )
            else:
                self._status[model_id] = ModelDownloadStatus(
                    model_id=model_id,
                    state="failed",
                    message="下载似乎没有完成，请重试。",
                    error="weight files missing after download",
                )
        except ModuleNotFoundError:
            self._status[model_id] = ModelDownloadStatus(
                model_id=model_id,
                state="failed",
                message="缺少 modelscope 组件，无法下载。请先安装 modelscope。",
                error="modelscope not installed",
            )
        except Exception as exc:  # pragma: no cover - network/disk dependent
            self._status[model_id] = ModelDownloadStatus(
                model_id=model_id,
                state="failed",
                message="下载失败了。常见原因是网络中断，请稍后重试。",
                error=str(exc)[-500:],
            )
        finally:
            self._tasks.pop(model_id, None)

    @staticmethod
    def _download_blocking(repo_id: str) -> None:
        from modelscope import snapshot_download

        snapshot_download(repo_id, cache_dir=str(MODELS_DIR))
