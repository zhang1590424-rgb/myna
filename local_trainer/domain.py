from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingDefaults(BaseModel):
    epochs: int = 3
    learning_rate: float = 0.0002
    lora_rank: int = 8
    batch_size: int = 2


class TemplatePreset(BaseModel):
    id: str
    title: str
    description: str
    goal_label: str
    sample_filename: str
    starter_prompt: str
    system_prompt: str
    defaults: TrainingDefaults = Field(default_factory=TrainingDefaults)
    sample_rows: list[dict[str, str]] = Field(default_factory=list)


class ModelOption(BaseModel):
    id: str
    name: str
    size_label: str
    parameter_count: str
    local_path: str | None = None
    available: bool = False
    recommended: bool = False
    note: str
    repo_id: str | None = None
    download_size_label: str | None = None


class ModelDownloadStatus(BaseModel):
    model_id: str
    state: Literal["idle", "downloading", "completed", "failed"] = "idle"
    progress: int = 0
    message: str = ""
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


class DatasetRecord(BaseModel):
    instruction: str
    output: str
    input: str = ""
    system: str | None = None


class DatasetUploadResult(BaseModel):
    dataset_id: str
    filename: str
    source_format: Literal["csv", "json", "jsonl", "xlsx"]
    training_format: Literal["alpaca"] = "alpaca"
    valid_count: int
    skipped_count: int
    warnings: list[str] = Field(default_factory=list)
    preview: list[DatasetRecord] = Field(default_factory=list)
    human_summary: str


class TrainingSettings(BaseModel):
    epochs: int = Field(default=3, ge=1, le=10)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    lora_rank: int = Field(default=8, ge=1, le=64)
    batch_size: int = Field(default=2, ge=1, le=16)


class TrainingPreset(BaseModel):
    id: str
    title: str
    description: str
    recommended: bool = False
    settings: TrainingSettings


class CreateTrainingJobRequest(BaseModel):
    template_id: str
    dataset_id: str
    model_id: str
    settings: TrainingSettings = Field(default_factory=TrainingSettings)


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    completed = "completed"
    failed = "failed"


class TrainingJob(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    template_id: str
    dataset_id: str
    model_id: str
    dataset_count: int
    settings: TrainingSettings
    status: JobStatus = JobStatus.pending
    progress: int = 0
    message: str = "等待开始"
    learned_count: int = 0
    loss: list[float] = Field(default_factory=list)
    output_dir: str | None = None
    error: str | None = None
    engine: str = "mock"
    pid: int | None = None
    run_dir: str | None = None
    eta: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class CompareRequest(BaseModel):
    job_id: str
    prompt: str


class CompareResponse(BaseModel):
    prompt: str
    before: str
    after: str
