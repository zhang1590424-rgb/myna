# 项目协作规则

## 项目定位

本地模型微调工具：面向不懂模型训练的 Mac 用户，用引导式 Web 界面完成本地 LoRA SFT 训练，训练前后对比看效果。

当前形态：**本机 Web 控制台**（`127.0.0.1:4180`），Python FastAPI 后端 + 静态前端，真实训练由 LLaMA-Factory 驱动。

## 目录结构

```
├── local_trainer/       # Python 后端服务（FastAPI）
│   ├── main.py          # 入口，API 路由
│   ├── domain.py        # Pydantic 领域模型
│   ├── dataset_manager.py  # 数据集管理（上传/校验/存储）
│   ├── data_validation.py  # CSV/JSON/JSONL/XLSX 解析与校验
│   ├── experiment_service.py  # 实验 CRUD、状态流转
│   ├── engine.py        # 训练引擎（真实/Mock 自动切换）
│   ├── queue_manager.py # 训练队列（单任务串行）
│   ├── inference_engine.py  # 对比推理（加载 base+LoRA 对话）
│   ├── compare.py       # 多实验对比（参数差异 + loss 曲线）
│   ├── llamafactory.py  # 生成 LLaMA-Factory 训练配置
│   ├── downloader.py    # ModelScope 模型下载
│   ├── model_registry.py    # 模型清单（基于 YAML）
│   ├── model_registry.yaml  # 支持的模型定义
│   ├── templates.py     # 场景模板 + 训练档位
│   ├── hardware.py      # 硬件检测（MPS/CUDA/CPU）
│   ├── environment.py   # 环境就绪检测
│   ├── persistence.py   # SQLite 持久化
│   ├── infer.py         # 推理子进程脚本
│   └── paths.py         # 路径常量
├── web/                 # 前端静态文件
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── scripts/             # 安装/检查/启动脚本
│   ├── install.command
│   ├── doctor.command
│   ├── start.command
│   └── verify.command
├── samples/             # 预留示例目录（示例数据现内置在 local_trainer/templates.py，按需实时生成 CSV）
├── tests/               # 自动化测试
├── runtime/             # 运行产物（自动生成，不提交）
│   ├── datasets/        # 上传的数据集
│   ├── runs/            # 训练任务产物
│   └── workbench.db     # SQLite 持久化
├── models/              # 下载的模型缓存（不提交）
├── LLaMA-Factory/       # 上游训练引擎（第三方依赖，不改）
├── PRODUCT.md           # 产品定位
├── DESIGN.md            # 视觉规范
├── INSTALL.md           # Agent 安装指南
├── requirements.txt     # Python 依赖
└── 启动Myna.command  # 双击启动入口
```

## 实现原则

1. **产品层和训练引擎隔离**：前端不拼训练参数，后端通过适配层生成 LLaMA-Factory 配置。
2. **小白优先**：用户可见文案说人话，专业参数默认折叠。
3. **不伪装真实训练**：环境未就绪时明确降级为演示模式，界面有标注。
4. **密钥不进代码**：不提交 token、私有下载地址、账号信息。
5. **不改 LLaMA-Factory 源码**：产品层通过适配器调用。

## 验证命令

```bash
# 测试 + lint
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m ruff check local_trainer tests

# 环境检查
scripts/doctor.command

# 启动服务
scripts/start.command
```

## 数据存储位置

| 数据 | 路径 | 持久性 |
|---|---|---|
| 上传的数据集 | `runtime/datasets/` | 服务重启后保留，可随时清理 |
| 下载的模型 | `models/` | 长期缓存，确认不需要才清 |
| 训练产物 | `runtime/runs/` | 服务重启后保留，可随时清理 |
| 实验元数据 | `runtime/workbench.db` | SQLite，可随时清理 |

## 清理规则

- `runtime/` 可随时清理，只放本机运行产物。
- 不清理 `models/`，除非确认对应模型不再需要。
- 不改动 `LLaMA-Factory/` 上游文件。

## Git 规则

`.gitignore` 已忽略 `.venv/`、`runtime/`、`models/`、`LLaMA-Factory/`、`m0_output/`。提交前检查：

```bash
git status --short
git diff --cached --stat
```

## 已踩平的坑（勿重走）

- 训练配置 `dataset_dir` 用**绝对路径**。
- **MPS/CPU 用 fp32**，仅 CUDA 用 bf16（`hardware.select_precision`）。
- 启动训练注入 `PYTORCH_ENABLE_MPS_FALLBACK=1`（`hardware.training_env`）。
- 进度读 `trainer_log.jsonl`，不解析 stdout。
- `llamafactory-cli` 用 `sys.executable` 旁路解析绝对路径，不依赖 PATH。
- `dataset_info.json` 的 `system` 列只在数据真含 system 时声明，否则 LLaMA-Factory 报 `KeyError: 'system'`。
- **多模态模型（如 Qwen3.5 的 `*ForConditionalGeneration`）推理必须用 `AutoModelForImageTextToText` 加载，不能用 `AutoModelForCausalLM`**。这类模型文本主干包在 `language_model` 子模块里，训练存出的 adapter key 带 `language_model` 前缀；用纯文本入口加载会让结构不一致，LoRA 权重全部 key 对不上、被 peft 静默丢弃，表现为「训练正常但测评对比前后无差异」。已在 `infer.py` 加 missing-keys 校验，加载不全直接报错而非静默。

## 项目状态

**M0–M3 全部完成，功能闭环可用。** 剩 M4 真人实测。

已实现能力：
- 本机 Web 控制台五步向导
- 模型下载（ModelScope 国内源）
- 数据上传与校验（CSV/JSON/JSONL/XLSX）
- 真实 LoRA SFT 训练（LLaMA-Factory）
- 实时训练进度与 loss 展示
- 训练前后推理对比
- LoRA adapter / 合并完整模型导出
- 参数档位（快/标准/精细）
- 环境未就绪自动降级演示模式

分发方式：Agent-assisted local install（不打 App 包）。

## 下一步

1. **M4 真人实测**：用户本人不看说明书跑一遍，记录卡点。
2. **训练参数校准**：按真机耗时校准三档默认值。
3. **跨机器验证**：干净 Mac 上只靠 `INSTALL.md` + `scripts/` 安装。
