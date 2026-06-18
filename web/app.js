"use strict";

/* ----------------------------------------------------------------------- *
 * 训练工作台 — vanilla JS SPA
 * 视图：experiments / new / detail / compare / models / datasets / lab
 * 状态用 hash 路由，数据全部走 /api/*。
 * ----------------------------------------------------------------------- */

const canvas = document.getElementById("canvas");
const toastEl = document.getElementById("toast");
const navButtons = [...document.querySelectorAll(".nav-item[data-view]")];
const queuePill = document.getElementById("queuePill");
const queueText = document.getElementById("queueText");

const state = {
  models: [],
  datasets: [],
  experiments: [],
  presets: [],
  templates: [],
  environment: null,
  filter: "all",
  searchTerm: "",
  datasetFilter: "all",
  datasetSearch: "",
  selectedForCompare: new Set(),
  pollTimer: null,
  downloadTimers: {},
  labResults: [],       // 缓存测评对比结果
  labPending: false,    // 是否有进行中的请求
  labStyle: "balanced", // 回答风格档位：steady / balanced / lively
  labAdvanced: {},      // 高级参数覆盖（留空则用档位默认）
};

/* ---------------- API helper ---------------- */
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = "请求失败，请稍后再试。";
    try {
      const payload = await res.json();
      detail =
        typeof payload.detail === "string"
          ? payload.detail
          : payload.detail?.message || detail;
    } catch {
      detail = res.statusText || detail;
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res;
}

/* ---------------- toast ---------------- */
let toastTimer = null;
function toast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.classList.toggle("error", isError);
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, 3600);
}

/* ---------------- small helpers ---------------- */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== null && v !== undefined && v !== false) {
      node.setAttribute(k, v);
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function clear(node) {
  node.replaceChildren();
}

function relativeTime(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  const diff = Date.now() - then;
  if (Number.isNaN(diff)) return "—";
  const sec = Math.round(diff / 1000);
  if (sec < 60) return "刚刚";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min} 分钟前`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr} 小时前`;
  const day = Math.round(hr / 24);
  return `${day} 天前`;
}

const STATUS_LABEL = {
  pending: "待运行",
  queued: "排队中",
  running: "运行中",
  stopping: "停止中",
  stopped: "已停止",
  completed: "完成",
  failed: "失败",
};

function modelName(id) {
  const m = state.models.find((x) => x.id === id);
  return m ? m.name : id;
}

function datasetName(id) {
  const d = state.datasets.find((x) => x.id === id);
  return d ? d.name : id;
}

function fmtLoss(value) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(4);
}

/* ---------------- data loading ---------------- */
async function refreshCore() {
  const [models, datasets, experiments] = await Promise.all([
    api("/api/models"),
    api("/api/datasets"),
    api("/api/experiments"),
  ]);
  state.models = models;
  state.datasets = datasets;
  state.experiments = experiments;
}

async function ensureMeta() {
  if (state.presets.length && state.templates.length) return;
  const [presets, templates, environment] = await Promise.all([
    api("/api/training-presets"),
    api("/api/templates"),
    api("/api/environment"),
  ]);
  state.presets = presets;
  state.templates = templates;
  state.environment = environment;
}

async function refreshQueue() {
  try {
    const q = await api("/api/queue");
    const busy = Boolean(q.running_id);
    queuePill.classList.toggle("busy", busy);
    if (busy) {
      queueText.textContent =
        q.queued_count > 1 ? `运行中 · 还有 ${q.queued_count - 1} 个排队` : "运行中";
    } else if (q.paused) {
      queueText.textContent = "队列已暂停";
    } else {
      queueText.textContent = "队列空闲";
    }
  } catch {
    /* ignore queue errors silently */
  }
}

/* whether any experiment is in a live (changing) state */
function hasLiveExperiment() {
  return state.experiments.some((e) =>
    ["queued", "running", "stopping", "pending"].includes(e.status)
  );
}

/* ---------------- router ---------------- */
function parseHash() {
  const raw = (location.hash || "#/experiments").slice(2);
  const [view, ...rest] = raw.split("/");
  return { view: view || "experiments", arg: rest.join("/") };
}

function navigate(path) {
  location.hash = path;
}

async function render() {
  stopPolling();
  const { view, arg } = parseHash();
  const navView = ["new", "detail", "compare"].includes(view) ? "experiments" : view;
  navButtons.forEach((b) => b.classList.toggle("active", b.dataset.view === navView));

  try {
    if (view === "experiments") {
      await ensureMeta();
      await refreshCore();
      renderExperiments();
      maybePoll();
    } else if (view === "new") {
      await ensureMeta();
      await refreshCore();
      // Engine mode can change after the user downloads a model, so re-check
      // the live environment instead of trusting the cached value.
      try {
        state.environment = await api("/api/environment");
      } catch {
        /* keep cached environment on failure */
      }
      renderNewExperiment(arg);
    } else if (view === "detail") {
      await ensureMeta();
      await refreshCore();
      await renderDetail(arg);
    } else if (view === "compare") {
      await refreshCore();
      await renderCompare(arg);
    } else if (view === "models") {
      state.models = await api("/api/models");
      renderModels();
    } else if (view === "datasets") {
      state.datasets = await api("/api/datasets");
      await ensureMeta();
      if (arg === "new") {
        renderDatasetUpload();
      } else {
        renderDatasets();
      }
    } else if (view === "lab") {
      await refreshCore();
      await renderLab();
    } else {
      navigate("/experiments");
    }
  } catch (err) {
    clear(canvas);
    canvas.appendChild(
      el("div", { class: "empty" }, [
        el("p", { class: "empty-title" }, "加载失败"),
        el("p", {}, err.message),
      ])
    );
    toast(err.message, true);
  }
  refreshQueue();
}

/* ---------------- polling (live experiments) ---------------- */
function maybePoll() {
  if (hasLiveExperiment()) {
    state.pollTimer = setTimeout(async () => {
      try {
        state.experiments = await api("/api/experiments");
        if (parseHash().view === "experiments") {
          renderExperiments();
        }
        refreshQueue();
      } catch {
        /* ignore */
      }
      maybePoll();
    }, 2500);
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

/* ===================================================================== *
 * VIEW: experiments list
 * ===================================================================== */
function statusDot(status) {
  return el("span", { class: `status-dot ${status}`, title: STATUS_LABEL[status] || status });
}

function experimentResult(exp) {
  if (exp.status === "running") {
    return el("span", {}, [
      el("span", { class: "inline-progress" }, [
        el("span", { style: `width:${exp.progress || 0}%` }),
      ]),
      el("span", { class: "tnum", style: "margin-left:8px" }, `${exp.progress || 0}%`),
    ]);
  }
  if (exp.status === "completed") {
    const finalLoss = exp.loss && exp.loss.length ? exp.loss[exp.loss.length - 1] : null;
    return el("span", {}, [
      `${exp.params.epochs} 轮 · loss `,
      el("span", { class: "loss-val" }, fmtLoss(finalLoss)),
    ]);
  }
  if (exp.status === "failed") return el("span", {}, "失败");
  if (exp.status === "queued") return el("span", {}, "排队中");
  if (exp.status === "stopped") return el("span", {}, "已停止");
  return el("span", {}, STATUS_LABEL[exp.status] || exp.status);
}

function filteredExperiments() {
  let list = state.experiments;
  if (state.filter === "sft") list = list.filter((e) => e.method === "sft");
  else if (state.filter === "dpo") list = list.filter((e) => e.method === "dpo");
  if (state.searchTerm) {
    const t = state.searchTerm.toLowerCase();
    list = list.filter(
      (e) =>
        e.name.toLowerCase().includes(t) ||
        modelName(e.model_id).toLowerCase().includes(t)
    );
  }
  return list;
}

function renderExperiments() {
  clear(canvas);

  const head = el("div", { class: "view-head" }, [
    el("div", {}, [
      el("h1", { class: "view-title" }, "实验"),
      el(
        "p",
        { class: "view-sub" },
        `${state.experiments.length} 条记录 · 模型 + 数据 + 参数的每一次组合`
      ),
    ]),
    el(
      "button",
      { class: "btn primary", onClick: () => navigate("/new") },
      "＋ 新建实验"
    ),
  ]);
  canvas.appendChild(head);

  const toolbar = el("div", { class: "view-toolbar" }, [
    el("div", { class: "filters" }, [
      filterBtn("all", "全部"),
      filterBtn("sft", "SFT"),
      filterBtn("dpo", "DPO"),
    ]),
    el("input", {
      class: "search",
      type: "search",
      placeholder: "搜索名称或模型…",
      value: state.searchTerm,
      onInput: (e) => {
        state.searchTerm = e.target.value;
        renderRows();
      },
    }),
  ]);
  canvas.appendChild(toolbar);

  const table = el("div", { class: "exp-table" });
  canvas.appendChild(table);

  function renderRows() {
    clear(table);
    const list = filteredExperiments();
    if (!list.length) {
      table.appendChild(
        el("div", { class: "empty" }, [
          el("p", { class: "empty-title" }, "还没有实验"),
          el(
            "p",
            {},
            state.experiments.length
              ? "没有匹配的实验，换个筛选条件试试。"
              : "点右上角「新建实验」，选模型、数据和参数，跑出第一条记录。"
          ),
        ])
      );
      return;
    }
    for (const exp of list) {
      table.appendChild(experimentRow(exp));
    }
  }
  renderRows();

  renderCompareBar();
}

function filterBtn(key, label) {
  return el(
    "button",
    {
      class: `filter ${state.filter === key ? "active" : ""}`,
      onClick: () => {
        state.filter = key;
        renderExperiments();
      },
    },
    label
  );
}

function experimentRow(exp) {
  const checked = state.selectedForCompare.has(exp.id);
  const check = el("input", {
    type: "checkbox",
    onClick: (e) => {
      e.stopPropagation();
      if (e.target.checked) state.selectedForCompare.add(exp.id);
      else state.selectedForCompare.delete(exp.id);
      renderCompareBar();
      row.classList.toggle("selected", e.target.checked);
    },
  });
  check.checked = checked;

  const row = el(
    "div",
    {
      class: `exp-row ${checked ? "selected" : ""}`,
      onClick: () => navigate(`/detail/${exp.id}`),
    },
    [
      el("div", { class: "exp-check" }, [check]),
      el("div", { class: "exp-main" }, [
        el("div", { class: "exp-name" }, [statusDot(exp.status), exp.name]),
        el(
          "div",
          { class: "exp-meta" },
          `${relativeTime(exp.created_at)} · ${datasetName(exp.dataset_id)} ${exp.dataset_count} 条`
        ),
      ]),
      el("div", { class: "exp-tags" }, [
        el("span", { class: "exp-tag-model" }, modelName(exp.model_id)),
        el("span", { class: "exp-method" }, exp.method),
      ]),
      el("div", { class: "exp-result" }, [experimentResult(exp)]),
    ]
  );
  return row;
}

function renderCompareBar() {
  let bar = document.getElementById("compareBar");
  const count = state.selectedForCompare.size;
  if (!bar) {
    bar = el("div", { class: "compare-bar", id: "compareBar" }, [
      el("span", { id: "compareCount" }),
      el(
        "button",
        {
          onClick: () => {
            const ids = [...state.selectedForCompare].join(",");
            navigate(`/compare/${ids}`);
          },
        },
        "对比"
      ),
      el(
        "button",
        {
          class: "clear",
          onClick: () => {
            state.selectedForCompare.clear();
            document
              .querySelectorAll(".exp-row.selected")
              .forEach((r) => r.classList.remove("selected"));
            document
              .querySelectorAll('.exp-check input[type="checkbox"]')
              .forEach((c) => (c.checked = false));
            renderCompareBar();
          },
        },
        "清除"
      ),
    ]);
    document.body.appendChild(bar);
  }
  const show = count >= 2;
  bar.classList.toggle("show", show);
  document.getElementById("compareCount").textContent = `已选 ${count} 个实验`;
}

/* ===================================================================== *
 * VIEW: new experiment (focus form) — also handles clone prefill
 * ===================================================================== */
function renderNewExperiment(cloneFromId) {
  clear(canvas);
  removeCompareBar();

  const availableModels = state.models.filter((m) => m.available);
  const source = cloneFromId
    ? state.experiments.find((e) => e.id === cloneFromId)
    : null;

  // form working state
  const standard = state.presets.find((p) => p.recommended) || state.presets[0];
  const form = {
    model_id: source ? source.model_id : availableModels[0]?.id || null,
    dataset_id: source ? source.dataset_id : null,
    method: source ? source.method : "sft",
    presetId: source ? null : standard?.id,
    params: source
      ? { ...source.params }
      : { ...(standard ? standard.params : {}) },
    name: "",
  };
  const sourceParams = source ? { ...source.params } : null;

  const view = el("div", { class: "form-view" });
  canvas.appendChild(view);
  view.appendChild(
    el("button", { class: "back-link", onClick: () => navigate("/experiments") }, "← 返回实验列表")
  );
  view.appendChild(
    el("h1", { class: "view-title" }, source ? "克隆实验" : "新建实验")
  );

  const page = el("div", { class: "form-page" });
  view.appendChild(page);

  if (state.environment && state.environment.engine === "mock") {
    page.appendChild(
      el("div", { class: "demo-banner" }, [
        el("strong", {}, "当前是演示模式，训练结果不是真实的。"),
        el(
          "span",
          {},
          "还缺训练组件或本地模型。请先到模型页下载模型，下载完成后无需重启，回到这里再发起训练就会自动启用真实训练。"
        ),
      ])
    );
  }

  if (source) {
    page.appendChild(
      el(
        "div",
        { class: "clone-hint" },
        `以「${source.name}」为模板。改动你想变的变量，其余保持一致，方便做对照实验。`
      )
    );
  }

  // ---- model field ----
  const modelChoices = el("div", { class: "choices" });
  function paintModels() {
    clear(modelChoices);
    if (!availableModels.length) {
      modelChoices.appendChild(
        el("p", { class: "muted" }, [
          "还没有可用的本地模型。先去 ",
          el("a", { href: "#/models" }, "模型页"),
          " 下载一个。",
        ])
      );
      return;
    }
    for (const m of state.models) {
      const usable = m.available;
      const choice = el(
        "button",
        {
          class: `choice ${form.model_id === m.id ? "active" : ""} ${usable ? "" : "disabled"}`,
          onClick: () => {
            if (!usable) {
              toast("这个模型还没下载，先去模型页下载它。", true);
              return;
            }
            form.model_id = m.id;
            paintModels();
          },
        },
        [
          el("span", { class: "choice-title" }, m.name),
          el(
            "span",
            { class: "choice-note" },
            usable ? m.parameter_count : "未下载"
          ),
        ]
      );
      modelChoices.appendChild(choice);
    }
  }
  paintModels();
  page.appendChild(fieldRow("底座模型", modelChoices));

  // ---- dataset field ----
  const datasetWrap = el("div", {});
  function compatibleDatasets() {
    const want = form.method === "dpo" ? "dpo_pairs" : "alpaca";
    return state.datasets.filter((d) => d.format === want);
  }
  function paintDataset() {
    clear(datasetWrap);
    const list = compatibleDatasets();
    if (!list.length) {
      datasetWrap.appendChild(
        el("p", { class: "muted" }, [
          form.method === "dpo"
            ? "没有偏好数据（chosen/rejected）。去 "
            : "没有问答数据。去 ",
          el("a", { href: "#/datasets" }, "数据页"),
          " 上传一份。",
        ])
      );
      form.dataset_id = null;
      return;
    }
    if (!list.find((d) => d.id === form.dataset_id)) {
      form.dataset_id = list[0].id;
    }
    const select = el(
      "select",
      {
        class: "input",
        onChange: (e) => {
          form.dataset_id = e.target.value;
        },
      },
      list.map((d) =>
        el(
          "option",
          { value: d.id, selected: d.id === form.dataset_id },
          `${d.name} · ${d.row_count} 条`
        )
      )
    );
    datasetWrap.appendChild(select);
  }
  paintDataset();
  page.appendChild(fieldRow("训练数据", datasetWrap));

  // ---- method field ----
  const methodChoices = el("div", { class: "choices" });
  function paintMethod() {
    clear(methodChoices);
    for (const [val, title, note] of [
      ["sft", "SFT", "监督微调，教模型按你的问答方式回答"],
      ["dpo", "DPO", "偏好优化，给出更好/更差两种回答精调风格"],
    ]) {
      methodChoices.appendChild(
        el(
          "button",
          {
            class: `choice ${form.method === val ? "active" : ""}`,
            onClick: () => {
              form.method = val;
              paintMethod();
              paintDataset();
            },
          },
          [
            el("span", { class: "choice-title" }, title),
            el("span", { class: "choice-note" }, note),
          ]
        )
      );
    }
  }
  paintMethod();
  page.appendChild(fieldRow("训练方法", methodChoices));

  // ---- params field (presets + advanced) ----
  const paramsWrap = el("div", {});
  const presetChoices = el("div", { class: "choices" });
  function paintPresets() {
    clear(presetChoices);
    for (const p of state.presets) {
      presetChoices.appendChild(
        el(
          "button",
          {
            class: `choice ${form.presetId === p.id ? "active" : ""}`,
            onClick: () => {
              form.presetId = p.id;
              form.params = { ...p.params };
              paintPresets();
              paintAdvanced();
            },
          },
          [
            el("span", { class: "choice-title" }, p.title),
            el("span", { class: "choice-note" }, p.description),
          ]
        )
      );
    }
  }
  paintPresets();

  const advanced = el("details", { class: "advanced" });
  const paramGrid = el("div", { class: "param-grid" });
  function paintAdvanced() {
    clear(advanced);
    advanced.appendChild(el("summary", {}, "展开调整高级参数"));
    clear(paramGrid);
    const specs = [
      ["epochs", "训练轮数", 1, 30, 1, "同样的数据反复学几遍。越多学得越透，但太多会死记硬背。"],
      ["learning_rate", "学习率", 0.00001, 0.01, 0.00001, "每次调整的步子大小。太大容易学跑偏，太小学得慢。"],
      ["lora_rank", "LoRA rank", 1, 64, 1, "给模型多大的“学习空间”。越大能学的越多，也更吃资源。"],
      ["batch_size", "批大小", 1, 16, 1, "一次同时看几条数据。越大越稳，但更占内存。"],
      ["grad_accum", "梯度累积步数", 1, 16, 1, "攒几批再更新一次模型。数据少就调小，模型学得更勤。"],
    ];
    if (form.method === "dpo") specs.push(["beta", "偏好强度 beta", 0.01, 1, 0.01, "多看重“更好/更差”的差距。越大越贴近你的偏好，太大易学偏。"]);
    for (const [key, label, min, max, step, hint] of specs) {
      const changed =
        sourceParams && Number(sourceParams[key]) !== Number(form.params[key]);
      const labelNode = el("label", {}, [
        label,
        changed
          ? el("span", { class: "param-changed" }, `（原 ${sourceParams[key]}）`)
          : null,
      ]);
      const input = el("input", {
        class: "input tnum",
        type: "number",
        min,
        max,
        step,
        value: form.params[key],
        onInput: (e) => {
          form.presetId = null;
          form.params[key] = Number(e.target.value);
          paintPresets();
          paintAdvanced();
        },
      });
      paramGrid.appendChild(
        el("div", {}, [labelNode, input, el("p", { class: "param-hint" }, hint)])
      );
    }
    advanced.appendChild(paramGrid);
  }
  paintAdvanced();
  if (source) advanced.open = true;

  paramsWrap.appendChild(presetChoices);
  paramsWrap.appendChild(advanced);
  page.appendChild(fieldRow("参数", paramsWrap));

  // ---- name field ----
  const nameInput = el("input", {
    class: "input",
    type: "text",
    placeholder: "留空将自动命名，如 sft-05b-客服-#3",
    onInput: (e) => {
      form.name = e.target.value;
    },
  });
  page.appendChild(fieldRow("名称（可选）", nameInput));

  // ---- actions ----
  const startBtn = el("button", { class: "btn primary" }, "开始训练 →");
  startBtn.addEventListener("click", async () => {
    if (!form.model_id) return toast("请先选一个可用的底座模型。", true);
    if (!form.dataset_id) return toast("请先选一份匹配的训练数据。", true);
    startBtn.disabled = true;
    try {
      await api("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: form.model_id,
          dataset_id: form.dataset_id,
          method: form.method,
          params: form.params,
          name: form.name.trim() || null,
          cloned_from: source ? source.id : null,
        }),
      });
      toast("实验已创建并加入队列。");
      state.selectedForCompare.clear();
      navigate("/experiments");
    } catch (err) {
      toast(err.message, true);
      startBtn.disabled = false;
    }
  });

  page.appendChild(
    el("div", { class: "form-actions" }, [
      el("button", { class: "btn ghost", onClick: () => navigate("/experiments") }, "取消"),
      startBtn,
    ])
  );
}

function fieldRow(label, body) {
  return el("div", { class: "field" }, [
    el("div", { class: "field-label" }, label),
    el("div", { class: "field-body" }, [body]),
  ]);
}

/* ===================================================================== *
 * VIEW: experiment detail
 * ===================================================================== */
async function renderDetail(id) {
  clear(canvas);
  removeCompareBar();
  const exp = await api(`/api/experiments/${id}`);

  canvas.appendChild(
    el("button", { class: "back-link", onClick: () => navigate("/experiments") }, "← 返回实验列表")
  );

  const actions = [];
  if (exp.status === "running" || exp.status === "queued") {
    actions.push(
      el(
        "button",
        {
          class: "btn",
          onClick: async () => {
            try {
              await api(`/api/experiments/${id}/stop`, { method: "POST" });
              toast("已请求停止训练。");
              render();
            } catch (err) {
              toast(err.message, true);
            }
          },
        },
        "停止"
      )
    );
  }
  if (exp.status === "completed") {
    actions.push(
      el("a", { class: "btn", href: `/api/experiments/${id}/export` }, "导出 LoRA"),
      el("a", { class: "btn", href: `/api/experiments/${id}/export?merge=true` }, "导出完整模型"),
      el("button", { class: "btn", onClick: () => navigate(`/lab`) }, "去测评试用")
    );
  }
  actions.push(
    el(
      "button",
      {
        class: "btn",
        onClick: () => navigate(`/new/${exp.id}`),
      },
      "以此为模板"
    ),
    el(
      "button",
      {
        class: "btn danger",
        onClick: async () => {
          if (!confirm("删除这个实验记录？训练产物也会被移除。")) return;
          try {
            await api(`/api/experiments/${id}`, { method: "DELETE" });
            toast("已删除。");
            navigate("/experiments");
          } catch (err) {
            toast(err.message, true);
          }
        },
      },
      "删除"
    )
  );

  canvas.appendChild(
    el("div", { class: "view-head" }, [
      el("div", {}, [
        el("h1", { class: "view-title" }, [statusDot(exp.status), " ", exp.name]),
        el(
          "p",
          { class: "view-sub" },
          `${STATUS_LABEL[exp.status] || exp.status} · 创建于 ${relativeTime(exp.created_at)}`
        ),
      ]),
      el("div", { class: "row-actions" }, actions),
    ])
  );

  if (exp.engine === "mock") {
    canvas.appendChild(
      el("div", { class: "demo-banner" }, [
        el("strong", {}, "这是演示模式的结果，不是真实训练。"),
        el(
          "span",
          {},
          "loss 曲线和产物都是模拟生成的，不能用于评估模型效果。请确认已下载本地模型，再重新发起一次训练即可得到真实结果（无需重启服务）。"
        ),
      ])
    );
  }

  if (exp.status === "running") {
    const pct = Math.round(exp.progress || 0);
    const indeterminate = pct <= 0;
    const bar = el(
      "div",
      { class: `train-progress${indeterminate ? " indeterminate" : ""}` },
      [el("span", { style: indeterminate ? "" : `width:${pct}%` })]
    );
    canvas.appendChild(
      el("div", { class: "train-progress-row" }, [
        bar,
        el("span", { class: "train-pct tnum" }, indeterminate ? "准备中" : `${pct}%`),
      ])
    );
    canvas.appendChild(el("p", { class: "muted" }, `${exp.message || ""} ${exp.eta ? "· 预计剩余 " + exp.eta : ""}`));
  }
  if (exp.status === "failed" && exp.error) {
    canvas.appendChild(
      el("p", { style: "color:var(--danger);margin:14px 0" }, `失败原因：${exp.error}`)
    );
  }

  const kv = el("dl", { class: "kv" }, [
    el("dt", {}, "模型"), el("dd", {}, modelName(exp.model_id)),
    el("dt", {}, "数据"), el("dd", {}, `${datasetName(exp.dataset_id)} · ${exp.dataset_count} 条`),
    el("dt", {}, "方法"), el("dd", {}, exp.method.toUpperCase()),
    el("dt", {}, "轮数"), el("dd", { class: "tnum" }, String(exp.params.epochs)),
    el("dt", {}, "学习率"), el("dd", { class: "tnum" }, String(exp.params.learning_rate)),
    el("dt", {}, "LoRA rank"), el("dd", { class: "tnum" }, String(exp.params.lora_rank)),
    el("dt", {}, "批大小"), el("dd", { class: "tnum" }, String(exp.params.batch_size)),
    exp.method === "dpo" ? el("dt", {}, "beta") : null,
    exp.method === "dpo" ? el("dd", { class: "tnum" }, String(exp.params.beta)) : null,
  ]);

  const left = el("div", {}, [
    el("div", { class: "section-label" }, "训练配置"),
    kv,
  ]);

  // loss chart
  const right = el("div", { class: "loss-panel" });
  if (exp.loss && exp.loss.length > 1) {
    const startLoss = exp.loss[0];
    const currentLoss = exp.loss[exp.loss.length - 1];

    right.appendChild(el("div", { class: "section-label" }, "训练效果"));

    // 曲线（flex:1 撑满，与左栏等高）
    const chartBox = el("div", { class: "loss-chart-box" }, [
      lossChart([{ name: exp.name, loss: exp.loss }], { detail: true }),
    ]);
    right.appendChild(chartBox);

    // 图下两个小字
    const footer = el("div", { class: "loss-foot" }, [
      el("div", { class: "loss-foot-item" }, [
        el("span", { class: "loss-foot-label" }, "初始 loss"),
        el("span", { class: "loss-foot-val" }, fmtLoss(startLoss)),
      ]),
      el("div", { class: "loss-foot-item" }, [
        el("span", { class: "loss-foot-label" }, "当前 loss"),
        el("span", { class: "loss-foot-val" }, fmtLoss(currentLoss)),
      ]),
    ]);
    right.appendChild(footer);
  } else {
    right.appendChild(el("div", { class: "section-label" }, "训练效果"));
    right.appendChild(el("p", { class: "muted" }, "训练完成后这里会显示 loss 下降曲线。"));
  }

  canvas.appendChild(el("div", { class: "detail-grid" }, [left, right]));

  // notes
  canvas.appendChild(el("div", { class: "section-label" }, "实验笔记"));
  const notes = el("textarea", {
    class: "input notes-area",
    placeholder: "跑完随手记：这次改了什么、效果如何、下次试什么…",
  });
  notes.value = exp.notes || "";
  const saveNotes = el("button", { class: "btn btn-sm", style: "margin-top:10px" }, "保存笔记");
  saveNotes.addEventListener("click", async () => {
    try {
      await api(`/api/experiments/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notes: notes.value }),
      });
      toast("笔记已保存。");
    } catch (err) {
      toast(err.message, true);
    }
  });
  canvas.appendChild(notes);
  canvas.appendChild(el("div", {}, [saveNotes]));

  // poll if live
  if (["running", "queued", "stopping", "pending"].includes(exp.status)) {
    state.pollTimer = setTimeout(() => {
      if (parseHash().view === "detail") renderDetail(id);
    }, 2500);
  }
}

/* simple inline SVG line chart for loss curves */
function lossChart(series, opts = {}) {
  const W = 320;
  const H = 140;
  const pad = 8;
  const all = series.flatMap((s) => s.loss);
  if (!all.length) return el("p", { class: "muted" }, "暂无数据");
  const maxV = Math.max(...all);
  const minV = Math.min(...all);
  const span = maxV - minV || 1;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", opts.detail ? "loss-chart loss-chart-detail" : "loss-chart");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");

  series.forEach((s, idx) => {
    if (!s.loss.length) return;
    const n = s.loss.length;
    const pts = s.loss.map((v, i) => {
      const x = pad + (i / Math.max(n - 1, 1)) * (W - pad * 2);
      const y = pad + (1 - (v - minV) / span) * (H - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", pts.join(" "));
    poly.setAttribute("fill", "none");
    const shade = series.length > 1 ? 0.27 + idx * 0.18 : 0.45;
    poly.setAttribute("stroke", `oklch(${shade} 0.07 250)`);
    poly.setAttribute("stroke-width", "2");
    poly.setAttribute("stroke-linejoin", "round");
    poly.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(poly);

    // 终点圆点（仅详情页单曲线）
    if (opts.detail && series.length === 1 && n > 1) {
      const lastX = pad + (W - pad * 2);
      const lastY = pad + (1 - (s.loss[n - 1] - minV) / span) * (H - pad * 2);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", lastX);
      dot.setAttribute("cy", lastY);
      dot.setAttribute("r", "3");
      dot.setAttribute("fill", `oklch(${shade} 0.07 250)`);
      dot.setAttribute("vector-effect", "non-scaling-stroke");
      svg.appendChild(dot);
    }
  });
  return svg;
}

/* ===================================================================== *
 * VIEW: compare
 * ===================================================================== */
async function renderCompare(idsArg) {
  clear(canvas);
  removeCompareBar();
  const ids = idsArg.split(",").filter(Boolean);
  const data = await api(`/api/experiments/compare?ids=${ids.join(",")}`);

  canvas.appendChild(
    el("button", { class: "back-link", onClick: () => navigate("/experiments") }, "← 返回实验列表")
  );
  canvas.appendChild(el("h1", { class: "view-title" }, "对比"));
  canvas.appendChild(
    el("p", { class: "view-sub" }, `${data.experiments.length} 个实验 · 相同的行已折叠，只看差异`)
  );

  // diff table
  const table = el("table", { class: "compare-table" });
  const headRow = el("tr", {}, [
    el("th", {}, ""),
    ...data.experiments.map((e) => el("th", {}, e.name)),
  ]);
  table.appendChild(el("thead", {}, [headRow]));
  const tbody = el("tbody", {});
  let sameCount = 0;
  for (const row of data.rows) {
    if (row.same) sameCount += 1;
    const tr = el("tr", { class: row.same ? "same hidden" : "" }, [
      el("td", { class: "row-label" }, row.label),
      ...row.cells.map((c) =>
        el("td", { class: row.same ? "" : "diff tnum" }, c === null ? "—" : String(c))
      ),
    ]);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  canvas.appendChild(table);

  if (sameCount) {
    let folded = true;
    const toggle = el(
      "button",
      { class: "fold-toggle" },
      `显示 ${sameCount} 个相同项`
    );
    toggle.addEventListener("click", () => {
      folded = !folded;
      tbody.querySelectorAll("tr.same").forEach((tr) => tr.classList.toggle("hidden", folded));
      toggle.textContent = folded ? `显示 ${sameCount} 个相同项` : `折叠 ${sameCount} 个相同项`;
    });
    canvas.appendChild(toggle);
  }

  // loss overlay
  const withLoss = data.loss_series.filter((s) => s.loss && s.loss.length > 1);
  if (withLoss.length) {
    canvas.appendChild(el("div", { class: "section-label" }, "Loss 曲线叠加"));
    canvas.appendChild(lossChart(withLoss));
    canvas.appendChild(
      el(
        "div",
        { style: "display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;font-size:12px" },
        withLoss.map((s, i) =>
          el("span", { class: "muted" }, [
            el("span", {
              style: `display:inline-block;width:10px;height:2px;vertical-align:middle;margin-right:6px;background:oklch(${0.27 + i * 0.18} 0.07 250)`,
            }),
            s.name,
          ])
        )
      )
    );
  }

  // side-by-side output: same prompt across completed experiments
  const completed = data.experiments.filter((e) => e.status === "completed");
  canvas.appendChild(el("div", { class: "section-label" }, "试同一句话"));
  if (completed.length < 1) {
    canvas.appendChild(el("p", { class: "muted" }, "等实验训练完成后，可以在这里用同一个问题对比各模型的回答。"));
    return;
  }
  const promptInput = el("input", {
    class: "input",
    type: "text",
    placeholder: "输入一句话，看各实验模型怎么回答…",
  });
  const runBtn = el("button", { class: "btn primary" }, "并排运行");
  const outputs = el("div", { class: "compare-outputs" });
  runBtn.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return toast("先输入一句话。", true);
    runBtn.disabled = true;
    clear(outputs);
    for (const exp of completed) {
      const slot = el("div", { class: "compare-output" }, [
        el("div", { class: "col-name" }, exp.name),
        el("div", { class: "answer muted" }, "生成中…"),
      ]);
      outputs.appendChild(slot);
      // sequential to respect hardware mutex
      try {
        await api("/api/lab/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ experiment_id: exp.id, use_adapter: true }),
        });
        const res = await api("/api/lab/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
        slot.querySelector(".answer").textContent = res.answer;
        slot.querySelector(".answer").classList.remove("muted");
      } catch (err) {
        slot.querySelector(".answer").textContent = `（失败：${err.message}）`;
      }
    }
    runBtn.disabled = false;
  });
  canvas.appendChild(el("div", { class: "compare-prompt-row" }, [promptInput, runBtn]));
  canvas.appendChild(outputs);
}

/* ===================================================================== *
 * VIEW: models
 * ===================================================================== */
function renderModels() {
  clear(canvas);
  removeCompareBar();
  canvas.appendChild(
    el("div", { class: "view-head" }, [
      el("div", {}, [
        el("h1", { class: "view-title" }, "模型"),
        el("p", { class: "view-sub" }, "本地底座模型。下载后才能用于实验。"),
      ]),
    ])
  );

  const list = el("div", { class: "list" });
  for (const m of state.models) {
    list.appendChild(modelRow(m));
  }
  canvas.appendChild(list);

  // 切回模型页时，恢复仍在后台下载的模型的进度条与轮询。
  for (const m of state.models) {
    if (!m.available && m.repo_id) {
      restoreDownload(m);
    }
  }
}

async function restoreDownload(model) {
  let s;
  try {
    s = await api(`/api/models/${model.id}/download`);
  } catch {
    return;
  }
  if (s.state !== "downloading") return;
  const row = canvas.querySelector(`[data-model-id="${model.id}"]`);
  if (!row) return;
  const actions = row.querySelector(".row-actions");
  clear(actions);
  attachProgress(model, actions, s.progress || 0);
}

function modelRow(m) {
  const actions = el("div", { class: "row-actions" });
  const row = el("div", { class: "list-row", "data-model-id": m.id }, [
    el("div", {}, [
      el("div", { class: "row-name" }, [
        m.name,
        m.recommended ? el("span", { class: "badge-ok" }, "推荐起步") : null,
      ]),
      el(
        "div",
        { class: "row-note" },
        `${m.parameter_count} · ${m.learning_value}${m.download_size_label ? " · " + m.download_size_label : ""}`
      ),
    ]),
    actions,
  ]);

  if (m.available) {
    actions.appendChild(el("span", { class: "badge-ok" }, "已就绪"));
  } else if (m.repo_id) {
    const dlBtn = el("button", { class: "btn btn-sm" }, "下载");
    dlBtn.addEventListener("click", () => startDownload(m, row, dlBtn));
    actions.appendChild(dlBtn);
  } else {
    actions.appendChild(el("span", { class: "muted" }, "暂不支持下载"));
  }
  return row;
}

async function startDownload(model, row, btn) {
  btn.disabled = true;
  btn.textContent = "准备中…";
  try {
    await api(`/api/models/${model.id}/download`, { method: "POST" });
  } catch (err) {
    toast(err.message, true);
    btn.disabled = false;
    btn.textContent = "下载";
    return;
  }
  const actions = row.querySelector(".row-actions");
  clear(actions);
  attachProgress(model, actions, 0);
}

// 渲染进度条并启动轮询；startDownload 和 restoreDownload 共用。
function attachProgress(model, actions, initial) {
  const prog = el("div", { class: "dl-progress" }, [
    el("span", { style: `width:${initial}%` }),
  ]);
  const pct = el("span", { class: "muted tnum" }, `${initial}%`);
  const speed = el("span", { class: "muted dl-speed" }, "");
  actions.appendChild(prog);
  actions.appendChild(pct);
  actions.appendChild(speed);

  clearInterval(state.downloadTimers[model.id]);
  state.downloadTimers[model.id] = setInterval(async () => {
    try {
      const s = await api(`/api/models/${model.id}/download`);
      prog.firstChild.style.width = `${s.progress || 0}%`;
      pct.textContent = `${s.progress || 0}%`;
      speed.textContent = s.speed ? `· ${s.speed}` : "";
      if (s.state === "completed") {
        clearInterval(state.downloadTimers[model.id]);
        toast(`${model.name} 下载完成。`);
        state.models = await api("/api/models");
        renderModels();
      } else if (s.state === "failed") {
        clearInterval(state.downloadTimers[model.id]);
        toast(`${model.name} 下载失败：${s.error || s.message}`, true);
        renderModels();
      }
    } catch {
      /* ignore poll error */
    }
  }, 1500);
}

/* ===================================================================== *
 * VIEW: datasets
 * ===================================================================== */
function renderDatasets() {
  clear(canvas);
  removeCompareBar();

  const head = el("div", { class: "view-head" }, [
    el("div", {}, [
      el("h1", { class: "view-title" }, "数据"),
      el(
        "p",
        { class: "view-sub" },
        `${state.datasets.length} 份数据集 · SFT 用问答数据，DPO 用偏好数据`
      ),
    ]),
    el(
      "button",
      { class: "btn primary", onClick: () => navigate("/datasets/new") },
      "＋ 上传数据"
    ),
  ]);
  canvas.appendChild(head);

  const toolbar = el("div", { class: "view-toolbar" }, [
    el("div", { class: "filters" }, [
      datasetFilterBtn("all", "全部"),
      datasetFilterBtn("alpaca", "问答 SFT"),
      datasetFilterBtn("dpo_pairs", "偏好 DPO"),
    ]),
    el("input", {
      class: "search",
      type: "search",
      placeholder: "搜索名称或文件…",
      value: state.datasetSearch,
      onInput: (e) => {
        state.datasetSearch = e.target.value;
        renderDatasetRows();
      },
    }),
  ]);
  canvas.appendChild(toolbar);

  const list = el("div", { class: "list" });
  canvas.appendChild(list);

  function renderDatasetRows() {
    clear(list);
    const data = filteredDatasets();
    if (!data.length) {
      // 空状态：新用户给示例引导，否则提示换筛选条件
      if (state.datasets.length) {
        list.appendChild(
          el("div", { class: "empty" }, [
            el("p", { class: "empty-title" }, "没有匹配的数据集"),
            el("p", {}, "换个筛选条件或搜索词试试。"),
          ])
        );
      } else {
        const empty = el("div", { class: "empty" }, [
          el("p", { class: "empty-title" }, "还没有数据集"),
          el("p", {}, "点右上角「上传数据」，传一份你自己的训练数据。"),
        ]);
        if (state.templates.length) {
          empty.appendChild(
            el("p", { class: "muted", style: "margin-top:14px" }, "没有数据？下载示例先跑通：")
          );
          empty.appendChild(
            el(
              "div",
              { class: "sample-links", style: "justify-content:center" },
              state.templates.map((t) =>
                el("a", { class: "btn btn-sm", href: `/api/sample-data/${t.id}` }, t.title)
              )
            )
          );
        }
        list.appendChild(empty);
      }
      return;
    }
    for (const d of data) {
      list.appendChild(datasetRow(d));
    }
  }
  renderDatasetRows();
}

function datasetFilterBtn(key, label) {
  return el(
    "button",
    {
      class: `filter ${state.datasetFilter === key ? "active" : ""}`,
      onClick: () => {
        state.datasetFilter = key;
        renderDatasets();
      },
    },
    label
  );
}

function filteredDatasets() {
  let list = state.datasets;
  if (state.datasetFilter === "alpaca") list = list.filter((d) => d.format !== "dpo_pairs");
  else if (state.datasetFilter === "dpo_pairs") list = list.filter((d) => d.format === "dpo_pairs");
  if (state.datasetSearch) {
    const t = state.datasetSearch.toLowerCase();
    list = list.filter(
      (d) =>
        d.name.toLowerCase().includes(t) ||
        (d.source_filename || "").toLowerCase().includes(t)
    );
  }
  return list;
}

/* ---- 上传数据：独立聚焦页（对齐新建实验的交互范式） ---- */
function renderDatasetUpload() {
  clear(canvas);
  removeCompareBar();

  const view = el("div", { class: "form-view" });
  canvas.appendChild(view);
  view.appendChild(
    el("button", { class: "back-link", onClick: () => navigate("/datasets") }, "← 返回数据列表")
  );
  view.appendChild(el("h1", { class: "view-title" }, "上传数据"));

  const page = el("div", { class: "form-page" });
  view.appendChild(page);

  // step 1: 先选数据类型（独立先决步骤，带人话解释）
  const formatOptions = [
    { value: "alpaca", title: "问答数据（SFT）", note: "一问一答的对话格式" },
    { value: "dpo_pairs", title: "偏好数据（DPO）", note: "同一个问题的好回答和差回答" },
  ];
  let selectedFormat = formatOptions[0].value;

  const typeChoices = el("div", { class: "choices type-choices" });
  function paintTypeChoices() {
    clear(typeChoices);
    for (const opt of formatOptions) {
      const card = el(
        "button",
        { class: "choice" + (opt.value === selectedFormat ? " active" : "") },
        [
          el("span", { class: "choice-title" }, opt.title),
          el("span", { class: "choice-note" }, opt.note),
        ]
      );
      card.addEventListener("click", () => {
        selectedFormat = opt.value;
        paintTypeChoices();
      });
      typeChoices.appendChild(card);
    }
  }
  paintTypeChoices();
  page.appendChild(el("div", { class: "section-label" }, "1 · 选数据类型"));
  page.appendChild(typeChoices);

  // step 2: 上传文件
  page.appendChild(el("div", { class: "section-label" }, "2 · 上传文件"));
  const fileInput = el("input", { type: "file", accept: ".csv,.json,.jsonl,.xlsx", hidden: true });
  const drop = el("div", { class: "dropzone" }, [
    el("span", {}, [
      "把文件拖到这里，或 ",
      el("span", { class: "pick", onClick: () => fileInput.click() }, "点击选择"),
    ]),
    el("span", { class: "dropzone-hint" }, "支持 CSV · JSON · JSONL · XLSX"),
  ]);
  drop.appendChild(fileInput);

  async function upload(file) {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("format", selectedFormat);
    try {
      const result = await api("/api/datasets", { method: "POST", body: fd });
      toast(result.human_summary || "数据集已上传。");
      state.datasets = await api("/api/datasets");
      navigate("/datasets");
    } catch (err) {
      toast(err.message, true);
    }
  }
  fileInput.addEventListener("change", () => upload(fileInput.files[0]));
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("drag");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    upload(e.dataTransfer.files[0]);
  });
  page.appendChild(drop);

  // 示例数据下载：上传场景下作为兜底入口
  if (state.templates.length) {
    page.appendChild(el("div", { class: "section-label" }, "没有数据？下载示例先跑通"));
    page.appendChild(
      el(
        "div",
        { class: "sample-links" },
        state.templates.map((t) =>
          el("a", { class: "btn btn-sm", href: `/api/sample-data/${t.id}` }, t.title)
        )
      )
    );
  }
}

function datasetRow(d) {
  return el("div", { class: "list-row" }, [
    el("div", {}, [
      el("div", { class: "row-name" }, [
        el("span", { class: "row-name-text", title: d.name }, d.name),
        el("span", { class: "badge-ok" }, d.format === "dpo_pairs" ? "偏好 DPO" : "问答 SFT"),
      ]),
      el(
        "div",
        { class: "row-note" },
        `${d.row_count} 条 · 来自 ${d.source_filename} · ${relativeTime(d.created_at)}`
      ),
    ]),
    el("div", { class: "row-actions" }, [
      el(
        "button",
        {
          class: "btn btn-sm",
          onClick: () => previewDataset(d.id),
        },
        "预览"
      ),
      el(
        "button",
        {
          class: "btn btn-sm danger",
          onClick: async () => {
            if (!confirm(`删除数据集「${d.name}」？`)) return;
            try {
              await api(`/api/datasets/${d.id}`, { method: "DELETE" });
              toast("已删除。");
              state.datasets = await api("/api/datasets");
              renderDatasets();
            } catch (err) {
              toast(err.message, true);
            }
          },
        },
        "删除"
      ),
    ]),
  ]);
}

async function previewDataset(id) {
  try {
    const data = await api(`/api/datasets/${id}`);
    const rows = data.preview || [];
    if (!rows.length) return toast("这份数据没有可预览的内容。");
    const cols = Object.keys(rows[0]);
    const table = el("table", { class: "preview-table" }, [
      el("thead", {}, [el("tr", {}, cols.map((c) => el("th", {}, c)))]),
      el(
        "tbody",
        {},
        rows.map((r) => el("tr", {}, cols.map((c) => el("td", {}, String(r[c] ?? "")))))
      ),
    ]);
    // render inline below list
    const existing = document.getElementById("previewBox");
    if (existing) existing.remove();
    const box = el("div", { id: "previewBox" }, [
      el("div", { class: "section-label" }, `预览：${data.info.name}（前 ${rows.length} 条）`),
      table,
    ]);
    canvas.appendChild(box);
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    toast(err.message, true);
  }
}

/* ===================================================================== *
 * VIEW: lab
 * ===================================================================== */

// 回答风格档位 → 底层解码参数（与后端 STYLE_PRESETS 保持一致）。
// 用户只选档位，专业参数默认折叠在“高级”里。
const LAB_STYLES = [
  { id: "steady", title: "稳重", note: "回答更稳定保守，适合客服、问答", params: { temperature: 0.5, top_p: 0.9, repetition_penalty: 1.15, no_repeat_ngram_size: 3 } },
  { id: "balanced", title: "平衡", note: "稳定与灵活兼顾，适合日常对话", params: { temperature: 0.7, top_p: 0.9, repetition_penalty: 1.15, no_repeat_ngram_size: 3 } },
  { id: "lively", title: "活泼", note: "更有创意发挥，适合角色扮演、创作", params: { temperature: 1.0, top_p: 0.95, repetition_penalty: 1.2, no_repeat_ngram_size: 3 } },
];

const LAB_PARAM_SPECS = [
  ["temperature", "随机性 temperature", 0, 2, 0.1, "越高回答越天马行空，越低越保守。太高会胡说，为 0 易复读。"],
  ["top_p", "候选范围 top_p", 0, 1, 0.05, "越小越保守，只在最靠谱的词里挑。"],
  ["repetition_penalty", "防复读力度", 1, 1.5, 0.05, "对已说过的词降权，越大越不容易重复。太大会凑怪词。"],
  ["no_repeat_ngram_size", "禁止重复词组长度", 2, 4, 1, "禁止连续几个词的组合重复出现。"],
];

// 构建“回答风格”选择器：三档预设 + 可展开的高级参数，复用训练表单同款交互。
function buildStyleControl() {
  const wrap = el("div", { class: "lab-style" });
  wrap.appendChild(el("div", { class: "field-label" }, "回答风格"));

  const presetChoices = el("div", { class: "choices" });
  const advanced = el("details", { class: "advanced" });
  const paramGrid = el("div", { class: "param-grid" });

  function currentStyleParams() {
    const preset = LAB_STYLES.find((s) => s.id === state.labStyle) || LAB_STYLES[1];
    return { ...preset.params, ...state.labAdvanced };
  }

  function paintPresets() {
    clear(presetChoices);
    for (const s of LAB_STYLES) {
      presetChoices.appendChild(
        el(
          "button",
          {
            class: `choice ${state.labStyle === s.id && !Object.keys(state.labAdvanced).length ? "active" : ""}`,
            onClick: () => {
              state.labStyle = s.id;
              state.labAdvanced = {};
              paintPresets();
              paintAdvanced();
            },
          },
          [
            el("span", { class: "choice-title" }, s.title),
            el("span", { class: "choice-note" }, s.note),
          ]
        )
      );
    }
  }

  function paintAdvanced() {
    clear(advanced);
    advanced.appendChild(el("summary", {}, "展开调整高级参数"));
    clear(paramGrid);
    const values = currentStyleParams();
    for (const [key, label, min, max, step, hint] of LAB_PARAM_SPECS) {
      const input = el("input", {
        class: "input tnum",
        type: "number",
        min,
        max,
        step,
        value: values[key],
        onInput: (e) => {
          state.labAdvanced[key] = Number(e.target.value);
          paintPresets();
        },
      });
      paramGrid.appendChild(
        el("div", {}, [el("label", {}, label), input, el("p", { class: "param-hint" }, hint)])
      );
    }
    advanced.appendChild(paramGrid);
  }

  paintPresets();
  paintAdvanced();
  wrap.appendChild(presetChoices);
  wrap.appendChild(advanced);
  return wrap;
}

async function renderLab() {
  clear(canvas);
  removeCompareBar();
  const [status, completed] = await Promise.all([
    api("/api/lab/status"),
    Promise.resolve(state.experiments.filter((e) => e.status === "completed")),
  ]);

  canvas.appendChild(
    el("div", { class: "view-head" }, [
      el("div", {}, [
        el("h1", { class: "view-title" }, "测评"),
        el("p", { class: "view-sub" }, "对比训练前后的效果：同一个问题，看看模型学到了什么。"),
      ]),
    ])
  );

  if (!completed.length) {
    canvas.appendChild(
      el("div", { class: "empty" }, [
        el("p", { class: "empty-title" }, "还没有可试用的实验"),
        el("p", {}, "先训练完成一个实验，再回来这里对话验证。"),
      ])
    );
    return;
  }

  // --- 顶部控制栏：选择实验 + 加载 ---
  const select = el(
    "select",
    { class: "input" },
    completed.map((e) =>
      el("option", { value: e.id, selected: status.experiment_id === e.id }, e.name)
    )
  );
  const loadBtn = el("button", { class: "btn primary btn-sm" }, "加载实验");
  const targetLine = el("div", { class: "lab-target" });
  const controlBar = el("div", { class: "lab-control" }, [
    el("div", { class: "lab-control-left" }, [select, loadBtn]),
    targetLine,
  ]);

  function paintTarget(s) {
    clear(targetLine);
    if (s.loaded) {
      targetLine.appendChild(el("span", { class: "lab-loaded-badge" }, `✓ 已加载「${s.experiment_name}」`));
    } else {
      targetLine.appendChild(el("span", { class: "muted" }, s.message || "未加载任何模型。"));
    }
  }
  paintTarget(status);

  loadBtn.addEventListener("click", async () => {
    loadBtn.disabled = true;
    try {
      const s = await api("/api/lab/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ experiment_id: select.value, use_adapter: true }),
      });
      paintTarget(s);
      state.labResults = [];
      clear(compareFlow);
      toast("已加载，可以开始对话。");
    } catch (err) {
      toast(err.message, true);
    }
    loadBtn.disabled = false;
  });

  // --- Tab 切换：即时对比 / 批量测试 ---
  const tabCompare = el("button", { class: "lab-tab active" }, "即时对比");
  const tabBatch = el("button", { class: "lab-tab" }, "批量测试");
  const tabs = el("div", { class: "lab-tabs" }, [tabCompare, tabBatch]);

  const panelCompare = el("div", { class: "lab-panel" });
  const panelBatch = el("div", { class: "lab-panel", style: "display:none" });

  tabCompare.addEventListener("click", () => {
    tabCompare.classList.add("active");
    tabBatch.classList.remove("active");
    panelCompare.style.display = "";
    panelBatch.style.display = "none";
  });
  tabBatch.addEventListener("click", () => {
    tabBatch.classList.add("active");
    tabCompare.classList.remove("active");
    panelBatch.style.display = "";
    panelCompare.style.display = "none";
  });

  // ====== 即时对比面板 ======
  const compareFlow = el("div", { class: "compare-flow" });
  const input = el("textarea", { class: "input", placeholder: "输入问题，看训练前后对比…", rows: 2 });
  const sendBtn = el("button", { class: "btn primary" }, state.labPending ? "生成中…" : "对比生成");
  if (state.labPending) sendBtn.disabled = true;
  const inputRow = el("div", { class: "chat-input" }, [input, sendBtn]);

  // 快捷测试问题
  const starters = state.templates.map((t) => t.starter_prompt).filter(Boolean);
  const promptHints = el("div", { class: "prompt-hints" });
  starters.forEach((p) => {
    promptHints.appendChild(
      el("button", { class: "prompt-hint-btn", onClick: () => { input.value = p; } }, p)
    );
  });

  panelCompare.appendChild(promptHints);
  panelCompare.appendChild(buildStyleControl());
  panelCompare.appendChild(compareFlow);
  panelCompare.appendChild(inputRow);

  // 恢复之前缓存的对比结果
  function renderCachedResult(item) {
    const turn = el("div", { class: "compare-turn" }, [
      el("div", { class: "compare-question" }, item.prompt),
      el("div", { class: "compare-answers" }, [
        el("div", { class: "compare-col" }, [
          el("div", { class: "compare-label" }, "训练前（Base）"),
          el("div", { class: item.pending ? "compare-answer thinking" : "compare-answer" },
            item.pending ? "生成中…" : (item.error || item.base_answer)),
        ]),
        el("div", { class: "compare-col" }, [
          el("div", { class: "compare-label accent" }, "训练后（Fine-tuned）"),
          el("div", { class: item.pending ? "compare-answer thinking" : "compare-answer" },
            item.pending ? "等待中…" : (item.error || item.finetuned_answer)),
        ]),
      ]),
    ]);
    compareFlow.appendChild(turn);
    return turn;
  }
  state.labResults.forEach((item) => renderCachedResult(item));

  async function sendCompare() {
    const prompt = input.value.trim();
    if (!prompt) return;
    input.value = "";
    sendBtn.disabled = true;
    sendBtn.textContent = "生成中…";
    state.labPending = true;

    const item = { prompt, base_answer: "", finetuned_answer: "", pending: true, error: null };
    state.labResults.push(item);
    const turn = renderCachedResult(item);
    turn.scrollIntoView({ behavior: "smooth", block: "end" });

    try {
      const res = await api("/api/lab/compare-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, style: state.labStyle, ...state.labAdvanced }),
      });
      item.base_answer = res.base_answer;
      item.finetuned_answer = res.finetuned_answer;
      item.pending = false;
      const cols = turn.querySelectorAll(".compare-answer");
      cols[0].textContent = res.base_answer;
      cols[0].classList.remove("thinking");
      cols[1].textContent = res.finetuned_answer;
      cols[1].classList.remove("thinking");
    } catch (err) {
      item.error = `（失败：${err.message}）`;
      item.pending = false;
      const cols = turn.querySelectorAll(".compare-answer");
      cols[0].textContent = item.error;
      cols[0].classList.remove("thinking");
      cols[1].textContent = item.error;
      cols[1].classList.remove("thinking");
    }
    state.labPending = false;
    sendBtn.disabled = false;
    sendBtn.textContent = "对比生成";
  }
  sendBtn.addEventListener("click", sendCompare);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) sendCompare();
  });

  // ====== 批量测试面板 ======
  const batchInfo = el("p", { class: "batch-info" }, "从训练数据中抽取问题，一键跑 base 与 fine-tuned 的对比。");
  const batchBtn = el("button", { class: "btn primary" }, "开始批量测试");
  const batchResults = el("div", { class: "batch-results" });
  panelBatch.appendChild(batchInfo);
  panelBatch.appendChild(batchBtn);
  panelBatch.appendChild(batchResults);

  batchBtn.addEventListener("click", async () => {
    batchBtn.disabled = true;
    batchBtn.textContent = "测试中，请稍候…";
    clear(batchResults);
    try {
      const res = await api("/api/lab/batch-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      res.results.forEach((item, idx) => {
        const row = el("div", { class: "batch-row" }, [
          el("div", { class: "batch-question" }, `${idx + 1}. ${item.prompt}`),
          el("div", { class: "compare-answers" }, [
            el("div", { class: "compare-col" }, [
              el("div", { class: "compare-label" }, "训练前"),
              el("div", { class: "compare-answer" }, item.base_answer),
            ]),
            el("div", { class: "compare-col" }, [
              el("div", { class: "compare-label accent" }, "训练后"),
              el("div", { class: "compare-answer" }, item.finetuned_answer),
            ]),
          ]),
        ]);
        batchResults.appendChild(row);
      });
    } catch (err) {
      batchResults.appendChild(el("p", { class: "text-error" }, `测试失败：${err.message}`));
    }
    batchBtn.disabled = false;
    batchBtn.textContent = "重新测试";
  });

  // --- 组装页面 ---
  canvas.appendChild(controlBar);
  canvas.appendChild(tabs);
  canvas.appendChild(panelCompare);
  canvas.appendChild(panelBatch);
}

/* ---------------- compare bar lifecycle ---------------- */
function removeCompareBar() {
  const bar = document.getElementById("compareBar");
  if (bar) bar.classList.remove("show");
}

/* ---------------- boot ---------------- */
navButtons.forEach((b) =>
  b.addEventListener("click", () => navigate(`/${b.dataset.view}`))
);
window.addEventListener("hashchange", render);
render();
