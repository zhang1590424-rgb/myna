from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Sample data (starter datasets the user can download)
# --------------------------------------------------------------------------- #
class TemplatePreset(BaseModel):
    id: str
    title: str
    description: str
    sample_filename: str
    starter_prompt: str
    sample_rows: list[dict[str, str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ModelOption(BaseModel):
    id: str
    name: str
    parameter_count: str
    learning_value: str
    lf_template: str
    local_path: str | None = None
    available: bool = False
    recommended: bool = False
    repo_id: str | None = None
    download_size_label: str | None = None


class ModelDownloadStatus(BaseModel):
    model_id: str
    state: Literal["idle", "downloading", "completed", "failed"] = "idle"
    progress: int = 0
    message: str = ""
    speed: str = ""
    error: str | None = None


class ModelStatus(BaseModel):
    id: str
    name: str
    available: bool
    note: str


class EnvironmentStatus(BaseModel):
    python_ok: bool
    llamafactory_ok: bool
    torch_mps_ok: bool
    model_status: list[ModelStatus]
    progress: int
    message: str
    engine: str = "not_ready"
    engine_label: str = "环境未就绪，暂不能训练"
    can_train: bool = False


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
DatasetFormat = Literal["alpaca", "dpo_pairs"]


class DatasetRecord(BaseModel):
    """Supervised (SFT) row."""

    instruction: str
    output: str
    input: str = ""
    system: str | None = None


class PreferenceRecord(BaseModel):
    """Preference (DPO) row."""

    instruction: str
    chosen: str
    rejected: str


class DatasetInfo(BaseModel):
    id: str
    name: str
    source_filename: str
    format: DatasetFormat
    row_count: int
    created_at: str


class DatasetUploadResult(BaseModel):
    dataset_id: str
    name: str
    filename: str
    source_format: Literal["csv", "json", "jsonl", "xlsx"]
    format: DatasetFormat
    valid_count: int
    skipped_count: int
    warnings: list[str] = Field(default_factory=list)
    preview: list[dict[str, str]] = Field(default_factory=list)
    human_summary: str
    # 上传后的硬规则诊断结果，复用 DiagnosticCard 让前端用同一套渲染
    diagnostics: list["DiagnosticCard"] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
TrainingMethod = Literal["sft", "dpo"]


class ExperimentParams(BaseModel):
    epochs: int = Field(default=10, ge=1, le=30)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    lora_rank: int = Field(default=16, ge=1, le=64)
    batch_size: int = Field(default=2, ge=1, le=16)
    grad_accum: int = Field(default=2, ge=1, le=16)  # 梯度累积步数，小数据用小值多走几步
    beta: float = Field(default=0.1, gt=0, le=1.0)  # DPO preference strength


class TrainingPreset(BaseModel):
    id: str
    title: str
    description: str
    recommended: bool = False
    params: ExperimentParams


class ExperimentStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    completed = "completed"
    failed = "failed"


class DiagnosticAction(BaseModel):
    """Actionable button attached to a diagnostic card."""

    label: str
    action: Literal["retrain", "goto_data", "goto_eval"]
    params: dict[str, object] = Field(default_factory=dict)


class DiagnosticCard(BaseModel):
    """A single diagnostic finding shown after training completes."""

    level: Literal["ok", "warn", "error"]
    title: str
    suggestion: str
    observation: str | None = None
    interpretation: str | None = None
    mechanism: str | None = None  # 背后机理：模型为什么会这样（教学性说明）
    how_to_tell: str | None = None  # 如何判断：当多个原因可能并存时帮用户分辨
    next_step: str | None = None
    evidence: str | None = None
    action: DiagnosticAction | None = None
    # topic 用于前端去重过滤：同一 topic 的 ok 卡若与 warn/error 主诊断同主题，会被折叠
    topic: Literal[
        "train_loss", "train_process", "eval_loss", "train_eval", "data", "dpo", "general"
    ] = "general"
    rank: int = 50


class Experiment(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    name: str
    method: TrainingMethod = "sft"
    model_id: str
    dataset_id: str
    dataset_count: int = 0
    params: ExperimentParams = Field(default_factory=ExperimentParams)
    status: ExperimentStatus = ExperimentStatus.pending
    progress: int = 0
    message: str = "等待开始"
    loss: list[float] = Field(default_factory=list)
    eval_loss: list[float] = Field(default_factory=list)  # 验证集 loss，用于观察是否可能训过头
    eta: str | None = None
    output_dir: str | None = None
    run_dir: str | None = None
    pid: int | None = None
    error: str | None = None
    engine: str = "llamafactory"
    cloned_from: str | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[DiagnosticCard] = Field(default_factory=list)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class CreateExperimentRequest(BaseModel):
    model_id: str
    dataset_id: str
    method: TrainingMethod = "sft"
    params: ExperimentParams = Field(default_factory=ExperimentParams)
    name: str | None = None
    cloned_from: str | None = None


class BatchVariableRequest(BaseModel):
    """Create N experiments that differ on one parameter."""

    model_id: str
    dataset_id: str
    method: TrainingMethod = "sft"
    base_params: ExperimentParams = Field(default_factory=ExperimentParams)
    variable: Literal["epochs", "learning_rate", "lora_rank", "batch_size", "beta"]
    values: list[float]
    name_prefix: str | None = None


class UpdateExperimentRequest(BaseModel):
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None


# --------------------------------------------------------------------------- #
# Lab (chat verification) + compare
# --------------------------------------------------------------------------- #
class LabLoadRequest(BaseModel):
    experiment_id: str
    use_adapter: bool = True


class LabChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 120


class LabChatResponse(BaseModel):
    prompt: str
    answer: str


class LabStatus(BaseModel):
    loaded: bool = False
    experiment_id: str | None = None
    experiment_name: str | None = None
    use_adapter: bool = True
    message: str = "未加载任何模型。"


class ComparePromptRequest(BaseModel):
    experiment_ids: list[str]
    prompt: str


class LabCompareChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 120
    style: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None


class LabCompareChatResponse(BaseModel):
    prompt: str
    base_answer: str
    finetuned_answer: str


class LabBatchTestRequest(BaseModel):
    prompts: list[str] = Field(default_factory=list)
    max_new_tokens: int = 120


class LabBatchTestItem(BaseModel):
    prompt: str
    base_answer: str
    finetuned_answer: str


class LabBatchTestResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    results: list[LabBatchTestItem]


class LabResult(BaseModel):
    id: str
    experiment_id: str
    experiment_name: str
    kind: Literal["compare", "batch", "chat"]
    prompt: str
    base_answer: str = ""
    finetuned_answer: str = ""
    created_at: str
    data: dict[str, Any] = Field(default_factory=dict)


# DatasetUploadResult 引用了下方定义的 DiagnosticCard，需要 rebuild 让前向引用生效
DatasetUploadResult.model_rebuild()
