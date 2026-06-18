const state = {
  step: 0,
  environment: null,
  templates: [],
  models: [],
  presets: [],
  selectedTemplateId: "customer_service",
  selectedModelId: null,
  selectedPresetId: "standard",
  dataset: null,
  job: null,
  pollTimer: null,
  downloadTimers: {},
  isComparing: false,
};

const panels = [...document.querySelectorAll(".panel")];
const steps = [...document.querySelectorAll(".step")];
const nextButton = document.querySelector("#nextButton");
const backButton = document.querySelector("#backButton");
const exportButton = document.querySelector("#exportButton");
const exportMergeButton = document.querySelector("#exportMergeButton");
const stopButton = document.querySelector("#stopButton");
const compareButton = document.querySelector("#compareButton");
const compareInput = document.querySelector("#compareInput");
const fileInput = document.querySelector("#fileInput");
const dropzone = document.querySelector("#dropzone");
const localAddress = document.querySelector(".local-address");
const refreshEnvironmentButton = document.querySelector("#refreshEnvironmentButton");

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = "请求失败，请稍后再试。";
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : payload.detail?.message || detail;
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  return response.json();
}

function activeTemplate() {
  return state.templates.find((item) => item.id === state.selectedTemplateId);
}

function activeModel() {
  return state.models.find((item) => item.id === state.selectedModelId);
}

function hasUsableModel() {
  return state.models.some((item) => item.available);
}

async function init() {
  if (localAddress) {
    localAddress.textContent = window.location.host || "127.0.0.1:4180";
  }
  const [environment, templates, models, presets] = await Promise.all([
    api("/api/environment"),
    api("/api/templates"),
    api("/api/models"),
    api("/api/training-presets"),
  ]);
  state.templates = templates;
  state.models = models;
  state.environment = environment;
  state.presets = presets;
  state.selectedModelId = models.find((item) => item.recommended && item.available)?.id || models.find((item) => item.available)?.id || models[0]?.id;
  state.selectedPresetId = presets.find((item) => item.recommended)?.id || presets[0]?.id || "standard";
  applyPreset(state.selectedPresetId);
  resetComparePrompt();
  renderTemplates();
  renderModels();
  renderEnvironment(environment);
  renderPresets();
  bindEvents();
  render();
}

function bindEvents() {
  steps.forEach((stepButton) => {
    stepButton.addEventListener("click", () => {
      const targetStep = Number(stepButton.dataset.step);
      if (targetStep <= state.step || canEnterStep(targetStep)) {
        state.step = targetStep;
        render();
      }
    });
  });

  nextButton.addEventListener("click", handleNext);
  refreshEnvironmentButton?.addEventListener("click", refreshEnvironment);
  backButton.addEventListener("click", () => {
    state.step = Math.max(0, state.step - 1);
    render();
  });

  exportButton.addEventListener("click", () => {
    if (state.job?.id) {
      window.location.href = `/api/training/jobs/${state.job.id}/export`;
    }
  });

  exportMergeButton.addEventListener("click", () => {
    if (state.job?.id) {
      window.location.href = `/api/training/jobs/${state.job.id}/export?merge=true`;
    }
  });

  stopButton.addEventListener("click", async () => {
    if (!state.job?.id) {
      return;
    }
    state.job = await api(`/api/training/jobs/${state.job.id}/stop`, { method: "POST" });
    renderJob();
  });

  compareButton.addEventListener("click", runCompare);

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (file) {
      uploadDataset(file);
    }
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      uploadDataset(file);
    }
  });

  ["epochsInput", "learningRateInput", "rankInput", "batchInput"].forEach((id) => {
    document.querySelector(`#${id}`).addEventListener("input", renderSummary);
  });
}

function canEnterStep(step) {
  if (step <= 0) {
    return true;
  }
  if (step === 1) {
    return hasUsableModel();
  }
  if (step === 2) {
    return Boolean(state.dataset);
  }
  if (step === 3) {
    return Boolean(state.job);
  }
  if (step === 4) {
    return state.job?.status === "completed";
  }
  return false;
}

async function handleNext() {
  if (state.step === 1 && !state.dataset) {
    setStatus("#datasetStatus", "请先上传数据，或者下载示例数据试一遍。", true);
    return;
  }
  if (state.step === 2) {
    await startTraining();
    return;
  }
  if (state.step === 3 && state.job?.status !== "completed") {
    return;
  }
  if (state.step < 4) {
    state.step += 1;
    render();
  }
}

function render() {
  panels.forEach((panel) => {
    panel.classList.toggle("show", Number(panel.dataset.panel) === state.step);
  });
  steps.forEach((stepButton) => {
    const step = Number(stepButton.dataset.step);
    stepButton.classList.toggle("active", step === state.step);
    stepButton.classList.toggle("done", step < state.step);
  });
  document.querySelector("#sampleLink").href = `/api/sample-data/${state.selectedTemplateId}`;
  renderSummary();
  renderActions();
  renderJob();
}

function renderActions() {
  backButton.style.visibility = state.step === 0 ? "hidden" : "visible";
  const canExport = state.step === 4 && state.job?.status === "completed";
  exportButton.hidden = !canExport;
  exportMergeButton.hidden = !canExport;
  nextButton.hidden = state.step === 4;

  const labels = {
    0: "继续上传数据",
    1: "下一步",
    2: "开始训练",
    3: "查看结果",
  };
  nextButton.textContent = labels[state.step] || "下一步";
  nextButton.disabled = (state.step === 0 && !hasUsableModel()) || (state.step === 3 && state.job?.status !== "completed");
}

function renderEnvironment(environment) {
  state.environment = environment;
  document.querySelector("#environmentMessage").textContent = environment.message;
  document.querySelector("#environmentProgress").style.width = `${environment.progress}%`;
  document.querySelector("#enginePill").textContent =
    environment.engine === "llamafactory" ? "真实训练已就绪" : "演示模式";
  const modelReady = environment.model_status.some((item) => item.available);
  document.querySelector("#environmentChecklist").innerHTML = [
    dependencyItem({
      title: "Python 本地服务",
      ok: environment.python_ok,
      detail: environment.python_ok ? "已通过当前服务启动。" : "请先运行安装脚本。",
      action: environment.python_ok ? "" : "复制安装命令",
      command: "scripts/install.command",
    }),
    dependencyItem({
      title: "LLaMA-Factory 训练组件",
      ok: environment.llamafactory_ok,
      detail: environment.llamafactory_ok ? "真实训练组件已安装。" : "缺少训练组件，会降级为演示模式。",
      action: environment.llamafactory_ok ? "" : "复制安装命令",
      command: "scripts/install.command",
    }),
    dependencyItem({
      title: "Apple GPU 加速",
      ok: environment.torch_mps_ok,
      optional: true,
      detail: environment.torch_mps_ok ? "已检测到 MPS，可用 Mac GPU 训练。" : "未检测到 MPS，真实训练会更慢。",
      action: "",
      command: "",
    }),
    dependencyItem({
      title: "本地基础模型",
      ok: modelReady,
      detail: modelReady ? "已有可用模型，可以继续上传数据。" : "请选择一个模型下载，第一次建议 0.5B。",
      action: "",
      command: "",
    }),
  ].join("");
  document.querySelectorAll("[data-copy-command]").forEach((button) => {
    button.addEventListener("click", () => copyCommand(button.dataset.copyCommand));
  });
  setStatus(
    "#environmentStatus",
    modelReady ? "环境检查完成。模型和数据都只保存在本机。" : "还没有可用模型，请先在右侧下载一个。",
    !modelReady,
  );
}

function dependencyItem({ title, ok, optional = false, detail, action, command }) {
  const stateClass = ok ? "ok" : optional ? "warn" : "missing";
  const stateText = ok ? "已就绪" : optional ? "可选优化" : "未就绪";
  const button = action
    ? `<button class="text-button" type="button" data-copy-command="${command}">${action}</button>`
    : "";
  return `
    <div class="dependency-item ${stateClass}">
      <span class="dependency-dot" aria-hidden="true">${ok ? "✓" : optional ? "!" : "×"}</span>
      <div>
        <strong>${title}</strong>
        <p>${detail}</p>
      </div>
      <em>${stateText}</em>
      ${button}
    </div>
  `;
}

async function copyCommand(command) {
  if (!command) {
    return;
  }
  try {
    await navigator.clipboard.writeText(command);
    setStatus("#environmentStatus", `已复制：${command}`);
  } catch {
    setStatus("#environmentStatus", `请在项目根目录运行：${command}`);
  }
}

async function refreshEnvironment() {
  setStatus("#environmentStatus", "正在重新检测。");
  const [environment, models] = await Promise.all([api("/api/environment"), api("/api/models")]);
  state.models = models;
  if (!activeModel()?.available) {
    state.selectedModelId =
      models.find((item) => item.recommended && item.available)?.id || models.find((item) => item.available)?.id || models[0]?.id;
  }
  renderModels();
  renderEnvironment(environment);
  renderSummary();
  renderActions();
}

function renderTemplates() {
  const grid = document.querySelector("#templateGrid");
  grid.innerHTML = state.templates
    .map((template) => {
      const selected = template.id === state.selectedTemplateId ? " selected" : "";
      const custom = template.id === "custom" ? " custom" : "";
      return `
        <button class="template-card${selected}${custom}" type="button" data-template-id="${template.id}">
          ${templateIcon(template.id)}
          <h2>${template.title}</h2>
          <p>${template.description}</p>
        </button>
      `;
    })
    .join("");
  grid.querySelectorAll("[data-template-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTemplateId = button.dataset.templateId;
      state.dataset = null;
      resetComparePrompt();
      renderTemplates();
      renderDatasetCard();
      renderDatasetHint();
      renderSummary();
      render();
    });
  });
  renderDatasetHint();
}

function renderModels() {
  const grid = document.querySelector("#modelGrid");
  grid.innerHTML = state.models
    .map((model) => {
      const selected = model.id === state.selectedModelId ? " selected" : "";
      const disabled = model.available ? "" : " disabled";
      const action = model.available
        ? `<span class="model-tag">${selected ? "当前使用" : "已就绪"}</span>`
        : `<button class="model-download-btn" type="button" data-download-id="${model.id}">下载 (${model.download_size_label || ""})</button>`;
      return `
        <div class="model-card${selected}${disabled}" data-model-id="${model.id}">
          <h2>${model.name}</h2>
          <p>${model.size_label} · ${model.parameter_count}</p>
          <p>${model.note}</p>
          <p class="model-download-status" id="downloadStatus-${model.id}"></p>
          ${action}
        </div>
      `;
    })
    .join("");
  grid.querySelectorAll("[data-model-id]").forEach((card) => {
    card.addEventListener("click", () => {
      const model = state.models.find((item) => item.id === card.dataset.modelId);
      if (!model?.available) {
        return;
      }
      state.selectedModelId = card.dataset.modelId;
      renderModels();
      renderSummary();
    });
  });
  grid.querySelectorAll("[data-download-id]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      startModelDownload(button.dataset.downloadId);
    });
  });
}

async function startModelDownload(modelId) {
  const statusNode = document.querySelector(`#downloadStatus-${modelId}`);
  try {
    const status = await api(`/api/models/${modelId}/download`, { method: "POST" });
    if (statusNode) {
      statusNode.textContent = status.message || "正在下载…";
    }
    pollModelDownload(modelId);
  } catch (error) {
    if (statusNode) {
      statusNode.textContent = error.message;
    }
  }
}

function pollModelDownload(modelId) {
  window.clearInterval(state.downloadTimers[modelId]);
  state.downloadTimers[modelId] = window.setInterval(async () => {
    let status;
    try {
      status = await api(`/api/models/${modelId}/download`);
    } catch {
      return;
    }
    const statusNode = document.querySelector(`#downloadStatus-${modelId}`);
    if (statusNode) {
      statusNode.textContent = status.message || "";
    }
    if (status.state === "completed" || status.state === "failed") {
      window.clearInterval(state.downloadTimers[modelId]);
      if (status.state === "completed") {
        const [environment, models] = await Promise.all([api("/api/environment"), api("/api/models")]);
        state.models = models;
        state.selectedModelId = modelId;
        renderModels();
        renderEnvironment(environment);
        renderSummary();
        renderActions();
      }
    }
  }, 1500);
}

function renderPresets() {
  const row = document.querySelector("#presetRow");
  if (!row) {
    return;
  }
  row.innerHTML = state.presets
    .map((preset) => {
      const selected = preset.id === state.selectedPresetId ? " selected" : "";
      return `
        <button class="preset-chip${selected}" type="button" data-preset-id="${preset.id}">
          <strong>${preset.title}</strong>
          <span>${preset.description}</span>
        </button>
      `;
    })
    .join("");
  row.querySelectorAll("[data-preset-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedPresetId = button.dataset.presetId;
      applyPreset(state.selectedPresetId);
      renderPresets();
      renderSummary();
    });
  });
}

function applyPreset(presetId) {
  const preset = state.presets.find((item) => item.id === presetId);
  if (!preset) {
    return;
  }
  document.querySelector("#epochsInput").value = preset.settings.epochs;
  document.querySelector("#learningRateInput").value = preset.settings.learning_rate;
  document.querySelector("#rankInput").value = preset.settings.lora_rank;
  document.querySelector("#batchInput").value = preset.settings.batch_size;
}

async function uploadDataset(file) {
  setStatus("#datasetStatus", "正在检查数据格式。");
  const form = new FormData();
  form.append("template_id", state.selectedTemplateId);
  form.append("file", file);

  try {
    state.dataset = await api("/api/datasets/validate", {
      method: "POST",
      body: form,
    });
    renderDatasetCard();
    renderSummary();
    setStatus("#datasetStatus", state.dataset.human_summary);
  } catch (error) {
    state.dataset = null;
    renderDatasetCard();
    setStatus("#datasetStatus", error.message, true);
  }
}

function renderDatasetCard() {
  const card = document.querySelector("#datasetCard");
  if (!state.dataset) {
    card.hidden = true;
    card.innerHTML = "";
    return;
  }
  card.hidden = false;
  card.innerHTML = `
    <div class="file-icon" aria-hidden="true">✓</div>
    <div class="file-meta">
      <strong>${state.dataset.filename}</strong>
      <span>${state.dataset.valid_count} 条有效数据 · ${state.dataset.training_format}</span>
    </div>
    <div class="file-ok">格式没问题</div>
  `;
}

function renderSummary() {
  const template = activeTemplate();
  const model = activeModel();
  if (!template || !model) {
    return;
  }
  const count = state.dataset?.valid_count || 0;
  document.querySelector("#trainingSummary").innerHTML = `
    将基于 <b>${model.name}</b>，用你的 <b>${count || "待上传"} 条</b>数据，训练一个<b>${template.goal_label}</b>。第一次建议先用标准档跑通。
  `;
}

function renderDatasetHint() {
  const template = activeTemplate();
  if (!template) {
    return;
  }
  const hints = {
    customer_service: "适合 question,answer 或 问题,回答：一列顾客问题，一列标准回复。",
    knowledge_qa: "适合 question,answer 或 问题,答案：一列知识问题，一列准确答案。",
    roleplay: "适合 question,answer：一列用户说的话，一列你希望 AI 使用的语气回答。",
    rewrite: "适合 question,answer：一列原文或改写要求，一列改写后的目标文本。",
    custom: "不确定也可以先用 question,answer。每行一条输入和理想输出。",
  };
  document.querySelector("#datasetHint").textContent = hints[template.id] || hints.custom;
  document.querySelector("#templateHelp").textContent = `${template.description}。示例数据和默认对比问题会跟着变化。`;
}

function resetComparePrompt() {
  const template = activeTemplate();
  if (template && compareInput) {
    compareInput.value = template.starter_prompt;
  }
}

async function startTraining() {
  if (!state.dataset) {
    setStatus("#confirmStatus", "还没有可训练数据，请先回到上一步上传。", true);
    return;
  }
  if (!activeModel()?.available) {
    setStatus("#confirmStatus", "还没有可用模型，请先回到准备环境页下载模型。", true);
    return;
  }
  setStatus("#confirmStatus", "正在创建训练任务。");
  const payload = {
    template_id: state.selectedTemplateId,
    dataset_id: state.dataset.dataset_id,
    model_id: state.selectedModelId,
    settings: readSettings(),
  };
  try {
    state.job = await api("/api/training/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.step = 3;
    setStatus("#confirmStatus", "");
    render();
    pollJob();
  } catch (error) {
    setStatus("#confirmStatus", error.message, true);
  }
}

function readSettings() {
  return {
    epochs: Number(document.querySelector("#epochsInput").value || 3),
    learning_rate: Number(document.querySelector("#learningRateInput").value || 0.0002),
    lora_rank: Number(document.querySelector("#rankInput").value || 8),
    batch_size: Number(document.querySelector("#batchInput").value || 2),
  };
}

function pollJob() {
  window.clearInterval(state.pollTimer);
  state.pollTimer = window.setInterval(async () => {
    if (!state.job?.id) {
      return;
    }
    state.job = await api(`/api/training/jobs/${state.job.id}`);
    renderJob();
    renderActions();
    if (["completed", "failed", "stopped"].includes(state.job.status)) {
      window.clearInterval(state.pollTimer);
      if (state.job.status === "completed") {
        await runCompare();
      }
    }
  }, 700);
}

function renderJob() {
  const job = state.job;
  if (!job) {
    document.querySelector("#jobPercent").textContent = "0%";
    document.querySelector("#jobMessage").textContent = "等待开始";
    document.querySelector("#jobProgress").style.width = "0%";
    document.querySelector("#jobFacts").innerHTML = "";
    return;
  }
  document.querySelector("#jobPercent").textContent = `${job.progress}%`;
  document.querySelector("#jobMessage").textContent = job.message;
  document.querySelector("#jobProgress").style.width = `${job.progress}%`;
  document.querySelector("#jobFacts").innerHTML = [
    `已学习 ${job.learned_count} / ${job.dataset_count} 条`,
    statusLabel(job.status),
  ]
    .map((text) => `<span class="fact-pill">${text}</span>`)
    .join("");
  stopButton.disabled = job.status !== "running";
  renderLoss(job.loss || []);
}

function renderLoss(losses) {
  const line = document.querySelector("#lossLine");
  const stats = document.querySelector("#lossStats");
  if (!losses.length) {
    line.setAttribute("points", "");
    stats.innerHTML = `
      <span><b>起始</b>暂无</span>
      <span><b>当前</b>暂无</span>
      <span><b>最低</b>暂无</span>
    `;
    return;
  }
  const max = Math.max(...losses);
  const min = Math.min(...losses);
  const range = Math.max(0.001, max - min);
  const points = losses
    .map((loss, index) => {
      const x = losses.length === 1 ? 0 : (index / (losses.length - 1)) * 420;
      const y = 16 + ((max - loss) / range) * 104;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  line.setAttribute("points", points);
  stats.innerHTML = `
    <span><b>起始</b>${formatLoss(losses[0])}</span>
    <span><b>当前</b>${formatLoss(losses[losses.length - 1])}</span>
    <span><b>最低</b>${formatLoss(min)}</span>
  `;
}

async function runCompare() {
  if (!state.job?.id) {
    return;
  }
  if (state.isComparing) {
    return;
  }
  const prompt = compareInput.value.trim();
  if (!prompt) {
    setStatus("#compareStatus", "先输入一句要对比的问题。", true);
    return;
  }
  state.isComparing = true;
  setCompareLoading(true);
  setStatus("#compareStatus", "正在生成训练前后回答，稍等一下。");
  try {
    const result = await api("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.job.id, prompt }),
    });
    document.querySelector("#beforeAnswer").textContent = result.before;
    document.querySelector("#afterAnswer").textContent = result.after;
    setStatus("#compareStatus", "");
  } catch (error) {
    setStatus("#compareStatus", error.message, true);
  } finally {
    state.isComparing = false;
    setCompareLoading(false);
  }
}

function setCompareLoading(isLoading) {
  compareInput.disabled = isLoading;
  compareButton.disabled = isLoading;
  compareButton.textContent = isLoading ? "正在生成…" : "对比回答";
  compareButton.setAttribute("aria-busy", String(isLoading));
  if (isLoading) {
    document.querySelector("#beforeAnswer").textContent = "正在生成训练前回答…";
    document.querySelector("#afterAnswer").textContent = "正在生成训练后回答…";
  }
}

function formatLoss(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "暂无";
  }
  return Number(value).toFixed(3);
}

function statusLabel(status) {
  const labels = {
    pending: "等待开始",
    running: "训练中",
    stopping: "停止中",
    stopped: "已停止",
    completed: "已完成",
    failed: "失败",
  };
  return labels[status] || status;
}

function setStatus(selector, text, isError = false) {
  const node = document.querySelector(selector);
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function templateIcon(templateId) {
  const icons = {
    customer_service: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/></svg>',
    roleplay: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 22a8 8 0 0 1 16 0"/></svg>',
    rewrite: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>',
    knowledge_qa: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    custom: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M5 12h14"/></svg>',
  };
  return icons[templateId] || icons.custom;
}

init().catch((error) => {
  document.querySelector("#environmentMessage").textContent = error.message;
});
