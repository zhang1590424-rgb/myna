# 项目协作规则

## 项目定位

本仓库用于开发「本地小白模型微调工具」：面向不懂模型训练的 Mac 用户，用引导式界面完成本地 LoRA SFT 训练，并通过训练前后对比看到效果。

## 目录约定

| 路径 | 放什么 | 约束 |
|---|---|---|
| `local_trainer/` | Python 本地服务、领域模型、数据校验、训练引擎适配 | 只写产品服务层，不直接改 LLaMA-Factory 源码 |
| `web/` | 本机 Web 控制台静态前端 | 界面只通过 `/api/*` 和本地服务通信；不要再套仿 Mac App 窗口壳 |
| `scripts/` | Agent 和用户可执行的安装、检查、启动、验证脚本 | 脚本需可重复执行，输出人话错误，不写入密钥 |
| `samples/` | 可下载示例数据 | 文件名用英文，内容可中文 |
| `tests/` | 本项目的自动化测试 | 优先覆盖数据校验、配置生成、任务状态 |
| `runtime/` | 本地运行产物、上传数据、训练任务 | 自动生成，不提交 |
| `docs/superpowers/specs/` | 产品设计与规格文档 | 规范变化先改文档，再改实现 |
| `prototype/` | 已确认的交互原型 | 作为参考，不作为生产代码继续堆功能 |
| `LLaMA-Factory/` | 上游训练引擎源码 | 视为第三方依赖，非必要不改 |
| `models/` | 本地模型缓存 | 权重文件不做无关整理 |

## 实现原则

1. **产品层和训练引擎隔离**：前端不拼训练参数，后端通过引擎适配层生成 LLaMA-Factory 配置。
2. **小白优先**：用户可见文案说人话，专业参数默认折叠。
3. **不伪装真实训练**：模拟训练只用于产品闭环验证，代码和界面必须明确可替换为真实引擎。
4. **风险优先**：真实 SFT、模型下载、桌面打包是高风险点，必须独立验证。
5. **密钥不进代码**：本项目不应提交 token、私有下载地址、账号信息。

## 验证命令

常规改动后运行：

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m ruff check local_trainer tests
```

Agent 友好分发验证：

```bash
scripts/doctor.command
scripts/verify.command
```

启动本地服务：

```bash
scripts/start.command
```

## 清理规则

- `runtime/` 可随时清理，里面只放本机运行产物。
- 不清理 `models/`，除非明确知道对应模型缓存不再需要。
- 不改动 `LLaMA-Factory/` 的上游文件，除非任务明确要求修引擎兼容性。

## 当前实现进展（2026-06-18 更新）

### 一句话现状

**M0 技术验证已通过；M1 真实训练链路已接入；M2 改为 Agent-assisted local install，不打 App 包；M3 产品化（模型下载、参数档位、错误兜底、merge 导出）已完成；前端已从仿 App 小窗调整为本机 Web 控制台。** 剩 M4 真人实测需用户本人来。

### 里程碑总览

| 里程碑 | 目标 | 状态 |
|---|---|---|
| **M0 技术验证** | 证明 Mac 本地能训练出有效果的模型 | ✅ **已完成** |
| **M1 产品闭环** | 五步向导 + 本地服务，小白能点完整流程 | ✅ **已完成，真实训练已接入** |
| **M2 分发验证** | Agent 帮用户安装，脚本启动本机 Web 服务 | ✅ **已完成脚本版分发协议**（不打 App 包） |
| **M3 产品化** | 模型下载、参数档位、错误兜底、导出 | ✅ **已完成** |
| **M4 小白实测** | 真人不看说明书跑通 | ⬜ 待用户本人实测 |

### M0 技术验证结果（已通过）

| 验证项 | 结果 |
|---|---|
| MPS（Apple GPU）可用 | ✅ M2 / 16GB |
| 本地 LoRA SFT 跑通 | ✅ Qwen2.5-0.5B，360 步，约 26 分钟 |
| 模型真在学（loss 下降） | ✅ 3.05 → 0.0018 |
| 训练前后对比有肉眼可见效果 | ✅ 自称从「Qwen/通义千问/阿里云」改写为「小白训练师/老张」 |
| 国内源下载模型 | ✅ ModelScope，0.5B 约 1 分钟 |

验证产物（仅供参考，**不要写死进产品实现**）：`m0_lora_sft.yaml`（训练配置）、`m0_compare.py`（前后对比推理脚本）、`m0_output/`（LoRA adapter）。

### 总体判断

当前项目已是 **本机 Web 控制台 + 真实训练闭环完整可用**：前端五步流程、本地 FastAPI 服务、数据校验、LLaMA-Factory 配置生成，以及**真实 LoRA SFT 训练、实时进度、训练前后真实推理对比、LoRA adapter / 完整模型导出**全部打通。环境未就绪时自动降级为 `MockTrainingEngine` 演示模式，并在界面明确标注。

当前分发判断：**不做 Tauri/Electron/PyInstaller App 包**。第一批其他用户大概率会让 Agent 帮忙安装，所以分发重点是 `INSTALL.md` + `scripts/*.command` 的可重复执行、可检查、可恢复，而不是隐藏命令行。

### 已完成

| 模块 | 路径 | 状态 |
|---|---|---|
| 项目规则 | `AGENTS.md` | 已建立目录约定、验证命令、清理规则 |
| 产品/设计上下文 | `PRODUCT.md`, `DESIGN.md` | 已固化设计稿中的产品定位和克制灰阶视觉规范 |
| 前端五步流程 | `web/index.html`, `web/styles.css`, `web/app.js` | 已实现准备环境、选方向、喂数据、确认、训练、对比；外壳已改为本机 Web 控制台 |
| 本地服务 | `local_trainer/main.py` | FastAPI 服务已可启动，默认端口 `127.0.0.1:4180` |
| 数据校验 | `local_trainer/data_validation.py`, `local_trainer/store.py` | 支持 CSV / JSON / JSONL / 基础 XLSX，转换为 Alpaca 风格数据 |
| 模板与模型清单 | `local_trainer/templates.py` | 已有客服、角色扮演、文本改写、知识问答、自定义；可识别本地 Qwen 0.5B |
| 环境检测 | `local_trainer/environment.py` | 可检测 `llamafactory`、PyTorch MPS、本地模型 |
| 训练引擎 | `local_trainer/engine.py` | 环境就绪时使用 `LlamaFactoryTrainingEngine`，否则降级 `MockTrainingEngine` |
| LLaMA-Factory 配置生成 | `local_trainer/llamafactory.py` | 可生成 `dataset_info.json`、训练数据和 `train.yaml` |
| Agent 安装协议 | `INSTALL.md`, `scripts/` | 已有安装、检查、启动、验证脚本；适合 Agent 帮其他用户落地 |
| 分发保护 | `.gitignore` | 已忽略 `.venv/`, `runtime/`, `models/`, `LLaMA-Factory/`, `m0_output/`, 日志等大文件/运行产物 |
| 测试 | `tests/` | 已覆盖数据校验和 LLaMA-Factory 配置生成 |
| 示例数据 | `samples/customer-service.csv` | 可用于快速上传体验 |

### M1 已接入真实训练（2026-06-18 完成）

页面点击「开始训练」后默认走真实引擎 `LlamaFactoryTrainingEngine`，端到端验证通过：

- **引擎工厂** `build_engine()`：环境就绪（装了 llamafactory + 有本地模型）→ 真实引擎；否则降级 `MockTrainingEngine`。`/api/environment` 返回 `engine`/`engine_label`，前端环境卡片显示「真实训练 / 演示模式」。
- **真实训练**：`asyncio.create_subprocess_exec` 启动 `llamafactory-cli train <config>`，记录 `pid`/`run_dir`/`output_dir`，后台 `_watch` 任务轮询 `trainer_log.jsonl` 回填 progress/loss/eta。
- **安全停止**：SIGTERM → 超时 10s → kill，状态置 `stopped`。
- **真实对比**：`infer.py` 子进程加载 base 与 base+LoRA，返回 before/after 真实回答。
- **真实导出**：`export()` 把 output 目录打包成 `lora-adapter-{job_id}.zip`，后端用 `FileResponse` 返回。

设计文档：`docs/superpowers/specs/2026-06-18-M1真实训练链路-design.md`（含实现结果与踩坑记录）。

### 已踩平的坑（接入时已处理，勿重走）

- 训练配置 `dataset_dir` 用**绝对路径**（`llamafactory.py` 已用 `str(dataset_dir)`）。
- **MPS/CPU 用 fp32**，仅 CUDA 用 bf16（`hardware.select_precision`，已不再写死 bf16）。
- 启动训练注入 `PYTORCH_ENABLE_MPS_FALLBACK=1`（`hardware.training_env`）。
- 进度读 `trainer_log.jsonl`，不解析 stdout（`parse_trainer_log`）。
- `llamafactory-cli` 按 `sys.executable` 旁路解析绝对路径，不依赖 PATH（`hardware.llamafactory_cli`）。
- `dataset_info.json` 的 `system` 列只在数据真含 system 时声明，否则 LLaMA-Factory 报 `KeyError: 'system'`。

### 当前分发与启动方式

推荐给其他用户的方式是 **Agent-assisted local install**。Agent 在项目根目录执行：

```bash
scripts/install.command
scripts/doctor.command
scripts/verify.command
scripts/start.command
```

用户也可以双击根目录：

```text
启动小白训练师.command
```

访问地址：

```text
http://127.0.0.1:4180
```

### Web 控制台 UI 决策（2026-06-18 完成）

原型曾按 App 小窗设计，但现在项目定位是本机 Web 服务，所以已采用 **方案 B：轻量 Web 化**：

- 去掉浏览器里的仿 Mac 标题栏、红黄绿圆点和居中固定小窗。
- 页面改为顶部服务栏 + 左侧训练流程 + 主内容区。
- 顶部显示「本机 Web 控制台」、当前地址、真实训练/演示模式。
- 五步向导继续保留，不改成复杂后台。
- loss 曲线等专业信息继续折叠，训练完成后的核心动作仍是「试用对比」。

设计文档：`docs/superpowers/specs/2026-06-18-Agent友好分发与Web控制台-design.md`。

### 下一位 Agent 的建议任务顺序（M2/M3/分发基础已完成）

1. **M4 真人实测**：由用户本人不看说明书走一遍五步流程，记录卡点。
2. **真实训练默认参数校准**：当前「标准」档 epochs=3，按真机耗时再校准三档（快/标准/精细）。
3. **脚本安装跨机器验证**：拿一台干净 Mac，让 Agent 只根据 `INSTALL.md` 和 `scripts/` 安装，记录缺 Python、git、网络、磁盘等卡点。
4. **可选 M2 进阶**：若未来真的要给陌生小白公开发布，再评估 Tauri/PyInstaller 完整打包（需解决 torch/llamafactory 数 GB 随包分发，是深坑，非必要不做）。

#### M2/M3 已交付内容（2026-06-18）

- **M2 一键启动**：`启动小白训练师.command`（双击启动服务+自动开浏览器+关窗即停+缺环境友好提示）；`requirements.txt` 固化产品层依赖（llamafactory 仍从本地 `LLaMA-Factory/` 源码装）。
- **Agent 友好分发**：`INSTALL.md` 和 `scripts/install.command` / `scripts/doctor.command` / `scripts/start.command` / `scripts/verify.command` 已建立安装、检查、启动、验证协议；根目录 `启动小白训练师.command` 转发到 `scripts/start.command`。
- **Web 控制台化**：`web/index.html` / `web/styles.css` / `web/app.js` 已去掉仿 App 窗口壳，改为本机 Web 控制台；桌面和移动宽度均用 Playwright 截图检查过，无横向溢出。
- **M3 模型下载**：`local_trainer/downloader.py`（ModelScope 国内源下载到 `MODELS_DIR`，线程池跑不阻塞事件循环，磁盘空间不足/缺 modelscope 有人话提示）；`paths.model_dir_for_repo()` 定位落点（点号转 `___`）；`templates.get_model_catalog()` 给 1.5B/3B 加 `repo_id` 并按本地文件判断 `available`。API：`POST/GET /api/models/{id}/download`。
- **M3 参数档位**：`templates.TRAINING_PRESETS`（快/标准/精细），`GET /api/training-presets`；前端确认页档位 chip，点选回填四个高级参数。
- **M3 merge 导出**：`engine.LlamaFactoryTrainingEngine._export_merged()` 调 `llamafactory-cli export` 合并 LoRA 为完整模型并打 zip；`export(job_id, merge=bool)`，API `GET /api/training/jobs/{id}/export?merge=true`。前端两个按钮「导出 LoRA / 导出完整模型」。
- **M3 错误兜底**：创建任务时校验模型已就绪、有效数据 ≥3 条，给人话提示。
- 测试：新增 `tests/test_model_catalog.py`（catalog/presets/downloader/路径布局），全套 18 测试 + ruff 通过。

### 重要风险和注意事项

- `LLaMA-Factory/`、`models/`、`runtime/`、`m0_output/` 很大或属于本地运行产物，**不要提交到 git**，尤其不要提交 `model.safetensors`。
- `.gitignore` 已忽略这些目录；如果 `git status --short --ignored` 仍看到异常，要先确认是不是手动强制添加过。
- 当前看到有 `git add` 进程尝试把 `LLaMA-Factory/` 和 `models/` 加进去。提交前必须检查：

```bash
git status --short
git diff --cached --stat
```

- 如果已有暂存内容包含模型或上游仓库，先清理暂存区，不要直接提交。
- `m0_*` 文件是技术验证产物，不等于产品实现；保留可参考，但不要把它写死进 Web 控制台或训练服务。
- 真实训练参数要控制耗时。当前 M0 是 360 steps，Mac 上能跑但时间不短；给小白产品默认值应更保守。
- 不要魔改 `LLaMA-Factory/` 源码。产品层通过适配器调用它。
