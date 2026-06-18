# Myna — 本地模型微调工具

面向不懂模型训练的 Mac 用户，用引导式 Web 界面完成本地 LoRA SFT 训练，训练前后对比看效果。

---

## For AI Agents: 一键部署指南

> **如果你是 AI Agent**，你的用户希望你帮他把这个项目在本机跑起来。按以下步骤执行即可。整个过程约 5-10 分钟，无需用户手动操作。

### 前置要求

| 条件 | 最低要求 | 检查方式 |
|---|---|---|
| 操作系统 | macOS（推荐 Apple Silicon M1/M2/M3/M4） | `uname -m` 应为 `arm64` |
| Python | 3.10+ | `python3 --version` |
| Git | 已安装 | `git --version` |
| 磁盘空间 | ≥ 20GB 可用 | `df -h .` |

如果 Python 未安装，先执行：`brew install python@3.11`

### 部署步骤

```bash
# 1. 克隆项目
git clone https://github.com/zhang1590424-rgb/myna.git
cd myna

# 2. 安装环境（创建虚拟环境 + 安装依赖 + 下载 LLaMA-Factory 训练引擎）
bash scripts/install.command

# 3. 预下载入门模型 Qwen3.5-0.8B（约 2GB），让用户打开界面就能直接用
#    必须用与界面相同的下载器和 models/ 目录，不要换成 huggingface-cli 等其他工具，否则界面识别不到
MODELSCOPE_DOWNLOAD_PARALLELS=4 .venv/bin/python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', cache_dir='models', max_workers=8)"

# 4. 健康检查（验证 Python/MPS/依赖/磁盘）
bash scripts/doctor.command

# 5. 运行测试（确认代码完整可用）
bash scripts/verify.command

# 6. 启动服务
bash scripts/start.command
```

> 第 3 步若下载失败，不影响部署是否成功，可告知用户跳过，之后在界面「模型」页面手动下载。

### 成功标志

- `install.command` 最后输出 `[install] 安装完成`
- `doctor.command` 无 `[FAIL]` 项（`[WARN]` 可忽略）
- `verify.command` 测试全部通过
- `start.command` 输出 `Uvicorn running on http://127.0.0.1:4180`，浏览器自动打开

**部署完成后，告诉用户打开 http://127.0.0.1:4180 即可使用。**

### 常见问题处理

| 现象 | 原因 | 解决 |
|---|---|---|
| `python3: command not found` | 未安装 Python | `brew install python@3.11` |
| install 时 git clone 失败 | GitHub 网络问题 | 重试，或设置代理 `git config --global http.proxy ...` |
| doctor 报 `MPS not available` | Intel Mac 或 PyTorch 版本问题 | 不影响使用，训练会自动降级为 CPU 模式（较慢） |
| doctor 报磁盘空间不足 | 模型 + 产物需要空间 | 清理磁盘后重试 |
| verify 测试失败 | 依赖安装不完整 | 重新运行 `bash scripts/install.command` |
| 端口 4180 被占用 | 其他进程占用 | `lsof -i :4180` 找到并关闭，再启动 |

### 日常使用（已部署后）

后续只需启动服务：

```bash
cd myna
bash scripts/start.command
```

---

## 产品简介

### 当前能力

- **五步向导**：选方向 → 上传数据 → 确认设置 → 训练 → 对比试用
- **模型下载**：从 ModelScope 国内源下载 Qwen3.5 系列小模型
- **数据校验**：支持 CSV / JSON / JSONL / XLSX 格式
- **真实训练**：调用 LLaMA-Factory 做 LoRA SFT，支持快速/标准/精细三档
- **对比推理**：训练前后同时推理对比效果
- **模型导出**：导出 LoRA adapter 或合并后的完整模型
- **自动降级**：环境未就绪时明确标注为演示模式，不伪装真实训练

### 技术架构

```
浏览器 (127.0.0.1:4180)
    ↓ HTTP
Python FastAPI 后端 (local_trainer/)
    ↓ 适配层
LLaMA-Factory (训练引擎)
    ↓
Apple MPS / CUDA / CPU
```

### 本地数据说明

| 目录 | 内容 | 可否删除 |
|---|---|---|
| `models/` | 下载的模型缓存 | 可删，再用时重新下载 |
| `runtime/datasets/` | 上传的训练数据 | 可删 |
| `runtime/runs/` | 训练任务产物 | 可删 |
| `runtime/workbench.db` | 实验元数据 | 可删 |
| `LLaMA-Factory/` | 第三方训练引擎 | 可删，重新 install 会恢复 |
| `.venv/` | Python 虚拟环境 | 可删，重新 install 会恢复 |

所有训练数据和模型都保存在用户本机，不上传到任何云端。
