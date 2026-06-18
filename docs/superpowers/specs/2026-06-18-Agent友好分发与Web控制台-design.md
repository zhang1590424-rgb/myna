# Agent 友好分发与 Web 控制台设计

## 背景

项目已经能在本机跑通真实 LoRA SFT、模型下载、训练前后对比和导出。下一步不是打 App 包，而是让其他 Mac 用户在 Agent 帮助下稳定安装、启动和验证。

同时，当前界面仍保留桌面 App 窗口壳。在浏览器访问 `127.0.0.1:4180` 时，这层仿 Mac 窗口会让产品显得被缩小了一层，不符合本地 Web 服务的使用语境。

## 目标

1. 让 Agent 可以按固定协议完成安装、检查、启动和验证。
2. 让普通用户只需要打开浏览器页面体验训练流程。
3. 把界面外壳从“桌面 App 小窗”调整为“本机 Web 控制台”。
4. 保留原有五步向导，不引入后台系统、账号、云同步或 App 打包。

## 非目标

- 不做 Tauri、Electron、PyInstaller 或签名公证。
- 不把 Python、Torch、LLaMA-Factory、模型权重打进单个安装包。
- 不改 LLaMA-Factory 源码。
- 不新增云端服务，不上传用户数据。

## 分发结构

新增 `scripts/` 目录，放置 Agent 和用户都可以直接执行的脚本：

| 文件 | 用途 |
|---|---|
| `scripts/install.command` | 创建 `.venv`、安装项目依赖、准备 LLaMA-Factory |
| `scripts/doctor.command` | 检查 Python、依赖、MPS、磁盘、模型状态 |
| `scripts/start.command` | 启动本地 FastAPI 服务并打开浏览器 |
| `scripts/verify.command` | 跑测试、ruff，并验证本地 API 可访问 |

根目录保留 `启动小白训练师.command`，作为用户可见入口，内部转发到 `scripts/start.command`。

## Web 控制台调整

页面保留左侧步骤和主任务区，但去掉仿 Mac 标题栏、窗口圆点、居中小窗、固定 720px 视觉高度。

新的外层结构：

1. 顶部服务栏：显示产品名、本机 Web 控制台、服务地址和训练引擎状态。
2. 左侧流程栏：继续承载 0 到 5 的训练步骤。
3. 主内容区：使用浏览器可用空间，主面板设置合理最大宽度。
4. 底部操作栏：仍然 sticky，但宽度和主内容区一致，不遮挡核心内容。

## 用户影响

- 用户看到的是一个本地控制台，而不是浏览器里套 App。
- Agent 可以直接执行脚本并根据输出判断下一步。
- 首次安装失败时，错误会落在脚本阶段，不会等到用户进入页面后才暴露。
- 模型权重和运行产物不会被误提交或误分发。

## 验证

实现后运行：

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m ruff check local_trainer tests
scripts/doctor.command
scripts/verify.command
```

并用浏览器检查 `http://127.0.0.1:4180`，确认首屏不是仿 App 小窗，训练流程仍可进入下一步。
