# 安装与启动

本项目推荐用 **Agent-assisted local install**：用户把项目放到本机后，让 Agent 执行安装、检查、启动和验证。用户只负责最后在浏览器里使用。

## 给 Agent 的标准流程

在项目根目录依次执行：

```bash
scripts/install.command
scripts/doctor.command
scripts/verify.command
scripts/start.command
```

如果只想启动已经安装好的环境，执行：

```bash
scripts/start.command
```

## 脚本职责

| 脚本 | 作用 |
|---|---|
| `scripts/install.command` | 创建 `.venv`，安装 Python 依赖，准备并安装 `LLaMA-Factory` |
| `scripts/doctor.command` | 检查 Python、核心依赖、MPS、磁盘空间、本地模型 |
| `scripts/verify.command` | 跑单元测试、ruff，并检查本地 API 是否可访问 |
| `scripts/start.command` | 启动 `127.0.0.1:4180`，自动打开浏览器 |

## 常见判断

| 现象 | 判断 | 下一步 |
|---|---|---|
| 缺少 `.venv` | 还没安装运行环境 | 运行 `scripts/install.command` |
| 缺少 `LLaMA-Factory` | 训练引擎未安装 | 运行安装脚本，或检查网络和 git |
| 未检测到 MPS | 可能不是 Apple Silicon，或 PyTorch 未启用 MPS | 可以用，但训练会慢 |
| 暂无本地模型 | 还没下载模型 | 打开页面后在“准备环境”里下载模型 |
| 磁盘空间不足 | 模型和训练产物可能写不下 | 清理磁盘，至少预留 8GB，建议 20GB 以上 |

## 本地数据边界

训练数据、模型缓存和训练产物都保存在这台 Mac 上：

- `models/`：模型缓存，不提交到 git。
- `runtime/`：上传数据和训练任务产物，可清理。
- `LLaMA-Factory/`：第三方训练引擎源码，不作为本项目代码提交。

不要提交 token、私有下载地址、账号信息或模型权重文件。
