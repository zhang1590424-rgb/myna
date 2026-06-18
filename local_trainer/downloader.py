from __future__ import annotations

import asyncio
import re
import shutil
import sys

from .domain import ModelDownloadStatus
from .paths import MODELS_DIR, model_dir_for_repo


class ModelDownloader:
    """Downloads models from ModelScope via a background subprocess.

    Spawns a separate Python process to run snapshot_download, captures stderr
    (where modelscope/tqdm writes progress), and updates progress in real time.
    This approach is more reliable than in-process threading: the download won't
    block the event loop, a crash won't take down the server, and the user sees
    actual percentage updates.
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
                progress=1,
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

            await self._download_subprocess(model_id, repo_id)

        except Exception as exc:
            self._status[model_id] = ModelDownloadStatus(
                model_id=model_id,
                state="failed",
                message="下载失败了。常见原因是网络中断，请稍后重试。",
                error=str(exc)[-500:],
            )
        finally:
            self._tasks.pop(model_id, None)

    async def _download_subprocess(self, model_id: str, repo_id: str) -> None:
        """Spawn a subprocess to download, parse tqdm progress from stderr."""
        script = (
            "import sys; "
            "from modelscope import snapshot_download; "
            f"snapshot_download('{repo_id}', cache_dir='{MODELS_DIR}'); "
            "print('__DONE__')"
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Read stderr in background to parse progress
        asyncio.create_task(self._watch_progress(model_id, repo_id, proc))

        await proc.wait()

        if proc.returncode != 0:
            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            error_text = stderr_bytes.decode(errors="replace")[-500:]

            if "ModuleNotFoundError" in error_text or "No module named" in error_text:
                self._status[model_id] = ModelDownloadStatus(
                    model_id=model_id,
                    state="failed",
                    message="缺少 modelscope 组件，无法下载。请先安装 modelscope。",
                    error="modelscope not installed",
                )
            else:
                self._status[model_id] = ModelDownloadStatus(
                    model_id=model_id,
                    state="failed",
                    message="下载失败了。常见原因是网络中断，请稍后重试。",
                    error=error_text,
                )
            return

        # Verify files are on disk
        target = model_dir_for_repo(repo_id)
        ok = (target / "config.json").exists() and (
            any(target.glob("*.safetensors")) or (target / "pytorch_model.bin").exists()
        )
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

    async def _watch_progress(
        self, model_id: str, repo_id: str, proc: asyncio.subprocess.Process
    ) -> None:
        """Parse tqdm percentage from stderr and update status."""
        # tqdm outputs lines like: " 45%|████      | 900M/2.00G [00:12<00:15, 73.2MB/s]"
        pct_pattern = re.compile(r"(\d{1,3})%\|")
        stderr = proc.stderr
        if stderr is None:
            return

        buffer = ""
        while True:
            chunk = await stderr.read(512)
            if not chunk:
                break
            buffer += chunk.decode(errors="replace")
            # tqdm uses \r for in-place updates, split on that
            lines = buffer.split("\r")
            buffer = lines[-1]  # keep incomplete tail
            for line in lines[:-1]:
                match = pct_pattern.search(line)
                if match:
                    pct = int(match.group(1))
                    # Clamp to 1-99 during download (100 means verified complete)
                    pct = max(1, min(pct, 99))
                    self._status[model_id] = ModelDownloadStatus(
                        model_id=model_id,
                        state="downloading",
                        progress=pct,
                        message=f"正在下载 {repo_id}（{pct}%）",
                    )
