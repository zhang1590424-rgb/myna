from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time

from .domain import ModelDownloadStatus
from .paths import MODELS_DIR, model_dir_for_repo

# 让单个大文件用多连接分片下载（modelscope 默认 1，即串行，是慢的主因）。
DOWNLOAD_PARALLELS = "4"
# modelscope 下载中的临时目录名，用于统计在途字节。
TEMP_FOLDER_NAME = "._____temp"


class ModelDownloader:
    """Downloads models from ModelScope via a background subprocess.

    Spawns a separate Python process to run snapshot_download, and computes
    overall progress from bytes-on-disk / total-remote-bytes. This is more
    reliable than parsing tqdm: a model has many files, and tqdm reports each
    file 0→100% separately, which makes a single shared bar jump backwards.
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
        """Spawn a subprocess to download; progress tracked by bytes on disk."""
        script = (
            "import sys; "
            "from modelscope import snapshot_download; "
            f"snapshot_download('{repo_id}', cache_dir=r'{MODELS_DIR}', max_workers=8); "
            "print('__DONE__')"
        )

        env = dict(os.environ)
        env["MODELSCOPE_DOWNLOAD_PARALLELS"] = DOWNLOAD_PARALLELS

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # 后台跟踪进度：优先按字节算整体进度，拿不到总大小时降级解析 tqdm。
        total_bytes = await asyncio.to_thread(self._remote_total_bytes, repo_id)
        stderr_tail = bytearray()
        progress_task = asyncio.create_task(
            self._watch_progress(model_id, repo_id, proc, total_bytes, stderr_tail)
        )

        await proc.wait()
        await progress_task  # 等它读完剩余 stderr 再判定结果

        if proc.returncode != 0:
            error_text = bytes(stderr_tail).decode(errors="replace")[-500:]

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

    def _remote_total_bytes(self, repo_id: str) -> int:
        """查询仓库所有文件的总字节数，作为整体进度的分母。失败返回 0。"""
        try:
            from modelscope.hub.api import HubApi

            files = HubApi().get_model_files(repo_id, recursive=True)
            total = 0
            for f in files:
                if f.get("Type") == "tree":
                    continue
                total += int(f.get("Size") or f.get("size") or 0)
            return total
        except Exception:
            return 0

    def _downloaded_bytes(self, repo_id: str) -> int:
        """统计已落盘字节，含已完成文件和临时目录中的在途分片。"""
        total = 0
        target = model_dir_for_repo(repo_id)
        roots = [target, target / TEMP_FOLDER_NAME, MODELS_DIR / TEMP_FOLDER_NAME]
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    @staticmethod
    def _format_speed(bytes_per_sec: float) -> str:
        """把字节/秒格式化成人类可读速率；负值或极小值归零。"""
        v = max(bytes_per_sec, 0)
        if v < 1024:
            return f"{v:.0f} B/s"
        if v < 1024**2:
            return f"{v / 1024:.0f} KB/s"
        return f"{v / 1024**2:.1f} MB/s"

    async def _watch_progress(
        self,
        model_id: str,
        repo_id: str,
        proc: asyncio.subprocess.Process,
        total_bytes: int,
        stderr_tail: bytearray,
    ) -> None:
        """整体进度优先按字节算；拿不到总大小时降级解析 tqdm 百分比。"""
        stderr = proc.stderr

        if total_bytes > 0:
            # 字节进度：单调递增，不会因为切换文件而回跳。
            # 同时起一个协程持续排空 stderr，防止 pipe 写满阻塞子进程。
            drain = asyncio.create_task(self._drain_stderr(stderr, stderr_tail))
            last_pct = 1
            last_done = 0
            last_t = time.monotonic()
            while proc.returncode is None:
                done = await asyncio.to_thread(self._downloaded_bytes, repo_id)
                now = time.monotonic()
                # 速率 = 本轮新增字节 / 间隔；掉到 0 用户即可判断是网络问题。
                speed = self._format_speed((done - last_done) / max(now - last_t, 1e-6))
                last_done, last_t = done, now
                pct = int(done * 100 / total_bytes)
                pct = max(last_pct, min(pct, 99))
                last_pct = pct
                self._status[model_id] = ModelDownloadStatus(
                    model_id=model_id,
                    state="downloading",
                    progress=pct,
                    message=f"正在下载 {repo_id}（{pct}%）",
                    speed=speed,
                )
                await asyncio.sleep(1.5)
            await drain
            return

        # 降级：解析 tqdm 的单文件百分比（无法反映整体，仅聊胜于无）。
        pct_pattern = re.compile(r"(\d{1,3})%\|")
        if stderr is None:
            return

        buffer = ""
        while True:
            chunk = await stderr.read(512)
            if not chunk:
                break
            stderr_tail.extend(chunk)
            del stderr_tail[:-2000]
            buffer += chunk.decode(errors="replace")
            lines = buffer.split("\r")
            buffer = lines[-1]
            for line in lines[:-1]:
                match = pct_pattern.search(line)
                if match:
                    pct = max(1, min(int(match.group(1)), 99))
                    self._status[model_id] = ModelDownloadStatus(
                        model_id=model_id,
                        state="downloading",
                        progress=pct,
                        message=f"正在下载 {repo_id}（{pct}%）",
                    )

    async def _drain_stderr(
        self, stderr: asyncio.StreamReader | None, stderr_tail: bytearray
    ) -> None:
        """持续读 stderr 防止 pipe 阻塞，只保留尾部用于报错。"""
        if stderr is None:
            return
        while True:
            chunk = await stderr.read(512)
            if not chunk:
                break
            stderr_tail.extend(chunk)
            del stderr_tail[:-2000]
