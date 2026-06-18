# 本地小白模型微调工具

面向非技术用户的本地 LoRA SFT 微调工具。当前形态是本机 Web 控制台：用户通过浏览器访问 `127.0.0.1:4180`，用五步向导完成模型下载、数据校验、真实训练、训练前后对比和导出。

## 当前能力

- 本机 Web 控制台：选方向、上传数据、确认设置、训练进度、对比试用。
- Python FastAPI 本地服务：模板、模型清单、环境状态、数据校验、任务状态。
- 数据校验：CSV、JSON、JSONL、基础 XLSX。
- 模型下载：通过 ModelScope 下载 Qwen2.5 系列小模型到本机。
- 真实训练：环境就绪时调用 LLaMA-Factory 做 LoRA SFT；未就绪时明确降级为演示模式。
- 导出：支持导出 LoRA adapter 和合并后的完整模型。

## 推荐安装方式

本项目推荐让 Agent 在用户 Mac 上安装：

```bash
scripts/install.command
scripts/doctor.command
scripts/verify.command
scripts/start.command
```

详细说明见 `INSTALL.md`。

## 启动

```bash
scripts/start.command
```

打开：

```text
http://127.0.0.1:4180
```

## 验证

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m ruff check local_trainer tests
```

## 重要说明

`models/`、`runtime/`、`LLaMA-Factory/` 都是本机运行相关目录，不提交到 git。真实训练耗时取决于 Mac 芯片、内存、模型大小和数据量。
