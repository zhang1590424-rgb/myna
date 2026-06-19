"use strict";

/* ----------------------------------------------------------------------- *
 * 训练工作台 — vanilla JS SPA
 * 视图：home / experiments / new / detail / compare / models / datasets / lab
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
  labFilter: "all",
  labExperimentFilter: "all",
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

function formatDateTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
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
  const raw = (location.hash || "#/home").slice(2);
  const [view, ...rest] = raw.split("/");
  return { view: view || "home", arg: rest.join("/") };
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
    if (view === "home") {
      await ensureMeta();
      await refreshCore();
      await renderHome();
      maybePoll();
    } else if (view === "experiments") {
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
      await renderLab(arg);
    } else {
      navigate("/home");
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
 * VIEW: home (adaptive dashboard)
 * ===================================================================== */
async function renderHome() {
  clear(canvas);
  removeCompareBar();

  if (!state.experiments.length) {
    renderHomeEmpty();
  } else {
    await renderHomeDashboard();
  }
}

function renderHomeEmpty() {
  canvas.appendChild(renderHomeBanner({
    title: "Myna",
    subtitle: "训练一次，看见不同",
  }));

  canvas.appendChild(renderHomeShortcuts());
  canvas.appendChild(renderHomeActivityLists([], []));
}

async function renderHomeDashboard() {
  const completed = state.experiments.filter((e) => e.status === "completed");
  const labHistory = completed.length ? await loadAllLabHistory(completed.slice(0, 10)) : [];

  canvas.appendChild(renderHomeBanner({
    title: "Myna",
    subtitle: "训练一次，看见不同",
  }));

  canvas.appendChild(renderHomeShortcuts());
  canvas.appendChild(renderHomeActivityLists(latestExperiments(state.experiments, 6), labHistory.slice(0, 6)));
}

function renderHomeBanner({ title, subtitle }) {
  return el("section", { class: "home-banner" }, [
    el("div", { class: "home-banner-main" }, [
      el("h1", { class: "home-banner-title" }, title),
      el("p", { class: "home-banner-subtitle" }, subtitle),
    ]),
    el("div", { class: "home-banner-actions" }, [
      el("a", {
        class: "home-banner-link",
        href: "https://bytedance.larkoffice.com/docx/MLZOdgEqWoT5zvxoLkQcZv7Nnjd",
        target: "_blank",
        rel: "noopener",
      }, "产品介绍"),
    ]),
  ]);
}

function renderHomeShortcuts() {
  const items = [
    ["新建实验", "选模型、选数据、开始训练", "/new"],
    ["准备数据", "上传或管理训练数据集", "/datasets/new"],
    ["新建测评", "对比训练前后回答差异", "/lab/new"],
  ];
  return el("section", { class: "home-section" }, [
    el("div", { class: "home-shortcuts" },
      items.map(([label, hint, route], index) =>
        el("button", { class: "home-shortcut", onClick: () => navigate(route) }, [
          el("span", { class: "home-shortcut-index tnum" }, String(index + 1)),
          el("span", { class: "home-shortcut-text" }, [
            el("span", { class: "home-shortcut-label" }, label),
            el("span", { class: "home-shortcut-hint" }, hint),
          ]),
        ])
      )
    ),
  ]);
}

function renderHomeActivityLists(experiments, labHistory) {
  return el("section", { class: "home-section home-activity" }, [
    el("div", { class: "home-activity-panel" }, [
      renderHomePanelHead("训练列表"),
      renderHomeTrainingList(experiments),
    ]),
    el("div", { class: "home-activity-panel" }, [
      renderHomePanelHead("测评列表"),
      renderHomeEvaluationList(labHistory),
    ]),
  ]);
}

function renderHomePanelHead(title) {
  return el("div", { class: "home-section-head" }, [
    el("div", {}, [
      el("div", { class: "home-block-title" }, title),
    ]),
  ]);
}

function renderHomeTrainingList(experiments) {
  if (!experiments.length) {
    return el("div", { class: "home-list-empty" }, [
      el("div", { class: "home-list-empty-title" }, "还没有训练记录"),
    ]);
  }
  return el("div", { class: "home-list" },
    experiments.map((exp) => el("button", {
      class: "home-list-row",
      onClick: () => navigate(`/detail/${exp.id}`),
    }, [
      el("span", { class: "home-list-status" }, [
        statusDot(exp.status),
      ]),
      el("span", { class: "home-list-title" }, exp.name),
      el("span", { class: "home-list-meta" }, STATUS_LABEL[exp.status] || exp.status),
    ]))
  );
}

function renderHomeEvaluationList(items) {
  if (!items.length) {
    return el("div", { class: "home-list-empty" }, [
      el("div", { class: "home-list-empty-title" }, "还没有测评记录"),
    ]);
  }
  return el("div", { class: "home-list" },
    items.map((item) => {
      const status = labResultStatus(item);
      const prompt = item.prompt || labResultItems(item)[0]?.prompt || "测评记录";
      const kind = item.kind === "batch" ? "批量" : item.kind === "chat" ? "对话" : "单问";
      return el("button", { class: "home-list-row", onClick: () => navigate("/lab") }, [
        el("span", { class: "home-list-status" }, [
          statusDot(status === "running" ? "running" : status === "failed" ? "failed" : "completed"),
        ]),
        el("span", { class: "home-list-title" }, prompt),
        el("span", { class: "home-list-meta" }, kind),
      ]);
    })
  );
}

function latestExperiments(experiments, limit) {
  return [...experiments]
    .sort((a, b) => latestExperimentTime(b).localeCompare(latestExperimentTime(a)))
    .slice(0, limit);
}

function latestExperimentTime(exp) {
  return exp.updated_at || exp.created_at || "";
}

function experimentResultText(exp) {
  if (exp.status === "completed" && exp.loss && exp.loss.length) {
    return `loss ${fmtLoss(exp.loss[exp.loss.length - 1])}`;
  }
  return STATUS_LABEL[exp.status] || exp.status;
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

  const canTrain = !state.environment || state.environment.can_train !== false;

  if (!canTrain) {
    page.appendChild(
      el("div", { class: "blocked-banner" }, [
        el("strong", {}, "还不能开始训练。"),
        el(
          "span",
          {},
          "缺少训练组件或本地模型。请先到模型页下载一个模型，环境就绪后即可发起真实训练（无需重启）。"
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
      ["lora_rank", "LoRA rank", 1, 64, 1, "给模型多大的「学习空间」。越大能学的越多，也更吃资源。"],
      ["batch_size", "批大小", 1, 16, 1, "一次同时看几条数据。越大越稳，但更占内存。"],
      ["grad_accum", "梯度累积步数", 1, 16, 1, "攒几批再更新一次模型。数据少就调小，模型学得更勤。"],
    ];
    if (form.method === "dpo") specs.push(["beta", "偏好强度 beta", 0.01, 1, 0.01, "多看重「更好/更差」的差距。越大越贴近你的偏好，太大易学偏。"]);
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
  if (!canTrain) {
    startBtn.disabled = true;
    startBtn.title = "环境未就绪，请先到模型页下载模型。";
  }
  startBtn.addEventListener("click", async () => {
    if (!canTrain) return toast("环境未就绪，请先到模型页下载模型。", true);
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
  const exp = await api(`/api/experiments/${id}`);
  const route = parseHash();
  if (route.view !== "detail" || route.arg !== id) return;

  clear(canvas);
  removeCompareBar();

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
      el("button", { class: "btn", onClick: () => navigate(`/lab/${exp.id}`) }, "去测评试用")
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

  if (exp.status === "failed" && exp.error) {
    canvas.appendChild(
      el("p", { style: "color:var(--danger);margin:14px 0" }, `失败原因：${exp.error}`)
    );
  }

  // 训练基本数据（静态信息，放在最前面作为上下文）
  const summaryItems = [
    ["模型", modelName(exp.model_id)],
    ["数据", `${datasetName(exp.dataset_id)} · ${exp.dataset_count} 条`],
    ["方法", exp.method.toUpperCase()],
    ["状态", STATUS_LABEL[exp.status] || exp.status],
    ["轮数", String(exp.params.epochs), true],
    ["学习率", String(exp.params.learning_rate), true],
    ["LoRA rank", String(exp.params.lora_rank), true],
    ["批大小", String(exp.params.batch_size), true],
  ];
  if (exp.method === "dpo") summaryItems.push(["beta", String(exp.params.beta), true]);
  const summaryGrid = el(
    "div",
    { class: "detail-summary-grid" },
    summaryItems.map(([label, value, numeric]) =>
      el("div", { class: "summary-field" }, [
        el("span", { class: "summary-label" }, label),
        el("span", { class: `summary-value${numeric ? " tnum" : ""}` }, value),
      ])
    )
  );
  canvas.appendChild(
    el("section", { class: "detail-section" }, [
      el("div", { class: "section-label" }, "训练基本数据"),
      summaryGrid,
    ])
  );

  // 进度条紧贴 loss 曲线（都是动态训练进展信息）
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

  // loss chart
  const effect = el("section", { class: "detail-section loss-panel" });
  const isRunning = ["running", "queued", "pending"].includes(exp.status);
  const hasLoss = exp.loss && exp.loss.length > 1;

  if (hasLoss) {
    const startLoss = exp.loss[0];
    const currentLoss = exp.loss[exp.loss.length - 1];
    const dropPct = startLoss > 0 ? ((startLoss - currentLoss) / startLoss * 100).toFixed(1) : "—";
    const hasEval = exp.eval_loss && exp.eval_loss.length > 1;

    // 标题行
    const headerRow = el("div", { style: "display:flex;justify-content:space-between;align-items:center" }, [
      el("div", { class: "section-label", style: "margin-bottom:0" }, "训练效果"),
    ]);
    effect.appendChild(headerRow);
    const insightText = lossInsight(exp.loss, hasEval ? exp.eval_loss : []);
    const insightClass = insightText.startsWith("⚠️") ? "loss-insight insight-warn"
      : insightText.startsWith("✅") ? "loss-insight insight-ok"
      : "loss-insight";
    effect.appendChild(
      el("p", { class: insightClass }, insightText)
    );

    // 左右双栏
    const chartNode = hasEval
      ? trainEvalLossChart(exp.loss, exp.eval_loss)
      : trainEvalLossChart(exp.loss, []);
    const chartBox = el("div", { class: "loss-chart-box" }, [chartNode]);
    // 图例
    const legend = hasEval
      ? el("div", { class: "loss-chart-legend" }, [
          el("span", { class: "train" }, [el("i", {}), "训练 loss"]),
          el("span", { class: "val" }, [el("i", {}), "验证 loss"]),
        ])
      : el("div", { class: "loss-chart-legend" }, [
          el("span", { class: "train" }, [el("i", {}), "训练 loss"]),
        ]);
    chartBox.appendChild(legend);
    const chartCol = el("div", { class: "loss-chart-col" }, [chartBox]);

    // 右侧指标卡片
    const minEvalLoss = hasEval ? Math.min(...exp.eval_loss.filter(v => typeof v === "number" && Number.isFinite(v))) : null;
    const metricsCol = el("div", { class: "loss-metrics-col" }, [
      el("div", { class: "loss-metric-card neutral" }, [
        el("div", { class: "loss-metric-label" }, "初始 Loss"),
        el("div", { class: "loss-metric-val" }, fmtLoss(startLoss)),
      ]),
      el("div", { class: "loss-metric-card positive" }, [
        el("div", { class: "loss-metric-label" }, "当前 Loss"),
        el("div", { class: "loss-metric-val" }, fmtLoss(currentLoss)),
      ]),
      el("div", { class: "loss-metric-card neutral" }, [
        el("div", { class: "loss-metric-label" }, "总步数"),
        el("div", { class: "loss-metric-val" }, String(exp.loss.length)),
      ]),
      hasEval
        ? el("div", { class: "loss-metric-card warn" }, [
            el("div", { class: "loss-metric-label" }, "验证最低"),
            el("div", { class: "loss-metric-val" }, fmtLoss(minEvalLoss)),
          ])
        : el("div", { class: "loss-metric-card neutral" }, [
            el("div", { class: "loss-metric-label" }, "下降幅度"),
            el("div", { class: "loss-metric-val" }, dropPct !== "—" ? `${dropPct}%` : "—"),
          ]),
    ]);

    const body = el("div", { class: "loss-body" }, [chartCol, metricsCol]);
    effect.appendChild(body);
  } else if (isRunning) {
    // 训练中，数据还不够画图
    effect.appendChild(el("div", { class: "section-label" }, "训练效果"));
    effect.appendChild(
      el("div", { class: "loss-placeholder" }, [
        el("div", { class: "loss-placeholder-pulse" }),
        el("span", {}, "训练进行中，曲线即将出现…"),
      ])
    );
  } else {
    effect.appendChild(el("div", { class: "section-label" }, "训练效果"));
    effect.appendChild(el("p", { class: "muted" }, "训练完成后这里会显示 loss 下降曲线。"));
  }
  canvas.appendChild(effect);

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
    state.pollTimer = setTimeout(async () => {
      const route = parseHash();
      if (route.view !== "detail" || route.arg !== id) return;
      try {
        await renderDetail(id);
      } catch {
        // Keep the current detail page visible if a transient refresh fails.
      }
    }, 2500);
  }
}

function lossInsight(trainLoss, valLoss = []) {
  const train = trainLoss.filter((v) => typeof v === "number" && Number.isFinite(v));
  const val = valLoss.filter((v) => typeof v === "number" && Number.isFinite(v));
  if (train.length < 2) return "数据点不足，训练完成后再看趋势。";

  const trainFirst = train[0];
  const trainLast = train[train.length - 1];
  const trainDropRatio = (trainFirst - trainLast) / trainFirst; // >0 means dropped
  const trainDropped = trainDropRatio > 0.1; // at least 10% drop
  const trainCollapsed = trainDropRatio > 0.75; // dropped >75%, near zero

  if (val.length > 1) {
    const valFirst = val[0];
    const valLast = val[val.length - 1];
    const minVal = Math.min(...val);
    const valDropRatio = (valFirst - valLast) / valFirst;
    const valFlat = Math.abs(valDropRatio) < 0.1; // val barely moved
    const valRose = valLast > minVal * 1.08; // val bounced back up

    // Case 1: Classic overfitting — train drops, val rises
    if (trainDropped && valRose) {
      return "⚠️ 过拟合：模型开始「背答案」——训练 loss 在降，但验证 loss 反而上升了。建议减少训练轮数，或增加训练数据量。";
    }
    // Case 2: Memorization overfitting — train collapsed to ~0, val flat
    if (trainCollapsed && valFlat) {
      return "⚠️ 过拟合：训练 loss 降到接近 0，但验证 loss 没有跟着降，说明模型把训练数据背住了，但没学到通用规律。建议增加训练数据量，或选择「快速」档减少训练轮数。";
    }
    // Case 3: Healthy — both dropping, gap reasonable
    if (trainDropped && valLast <= valFirst && !valFlat) {
      return "✅ 训练健康：两条线都在下降，模型在学到有效规律。可以去「试用对比」验证实际效果。";
    }
    // Case 4: Train dropped, val flat but train didn't collapse — mild overfitting
    if (trainDropped && valFlat) {
      return "⚠️ 轻度过拟合：训练 loss 在降，但验证 loss 基本没变化，模型对新数据的泛化有限。建议增加训练数据量，让模型有更多样本可学。";
    }
    // Case 5: Neither moved much — underfitting
    if (!trainDropped && valFlat) {
      return "⚠️ 欠拟合：两条线都没有明显下降，模型似乎没有学进去。建议检查数据质量，或尝试「精细」档增加训练量。";
    }
    // Fallback with val
    return "训练 loss 在下降，验证 loss 走势不太典型。建议去「试用对比」看实际回答效果来判断。";
  }

  // No val loss available
  if (trainCollapsed) return "训练 loss 降到接近 0，模型可能已记住全部训练数据。建议去「试用对比」检查实际效果，如果回答没变化，考虑增加数据量。";
  if (trainDropped) return "训练 loss 在下降，模型正在学习这批数据的模式。训练完成后去「试用对比」验证效果。";
  return "训练 loss 暂未明显下降，模型可能还没学进去。建议检查数据是否格式正确，或尝试增加训练轮数。";
}

/* simple inline SVG line chart for loss curves */
function lossChart(series, opts = {}) {
  const W = opts.detail ? 920 : 320;
  const H = opts.detail ? 176 : 140;
  const pad = opts.detail ? 0 : 8;
  const all = series.flatMap((s) => s.loss);
  if (!all.length) return el("p", { class: "muted" }, "暂无数据");
  const maxV = Math.max(...all);
  const minV = Math.min(...all);
  const span = maxV - minV || 1;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", opts.detail ? "loss-chart loss-chart-detail" : "loss-chart");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

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

/* 双 loss 曲线：训练 loss（蓝，全程）+ 验证 loss（橙，按 epoch）。
   升级版：网格线 + 渐变填充 + 终点标记，SVG 自适应容器。 */
function trainEvalLossChart(trainLoss, valLoss) {
  const W = 640, H = 220;
  const padL = 16, padR = 16, padT = 16, padB = 20;
  const xw = W - padL - padR, yh = H - padT - padB;
  const all = [...trainLoss, ...(valLoss || [])].filter((v) => typeof v === "number");
  if (!all.length) return el("p", { class: "muted" }, "暂无曲线数据");
  const maxV = Math.max(...all);
  const minV = Math.min(...all);
  const span = maxV - minV || 1;
  const Y = (v) => padT + (1 - (v - minV) / span) * yh;
  const xOf = (i, n) => padL + (n <= 1 ? 0 : (i / (n - 1)) * xw);

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "loss-chart-detail");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const add = (tag, attrs) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    svg.appendChild(node);
    return node;
  };

  // 渐变定义
  const defs = document.createElementNS(ns, "defs");
  defs.innerHTML = `<linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#5E6AD2" stop-opacity="0.1"/>
    <stop offset="100%" stop-color="#5E6AD2" stop-opacity="0"/>
  </linearGradient>`;
  svg.appendChild(defs);

  // 水平网格线（4 条）
  const gridCount = 4;
  for (let i = 0; i <= gridCount; i++) {
    const gy = padT + (i / gridCount) * yh;
    add("line", { x1: padL, y1: gy, x2: W - padR, y2: gy, stroke: "#e8e8f0", "stroke-width": "0.5" });
  }

  // 训练 loss 路径
  const path = (arr) =>
    arr
      .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i, arr.length).toFixed(1)},${Y(v).toFixed(1)}`)
      .join(" ");

  if (trainLoss.length > 1) {
    const trainPath = path(trainLoss);
    // 渐变填充区域
    const lastX = xOf(trainLoss.length - 1, trainLoss.length);
    const firstX = xOf(0, trainLoss.length);
    const baseY = padT + yh;
    add("path", {
      d: `${trainPath} L${lastX.toFixed(1)},${baseY} L${firstX.toFixed(1)},${baseY} Z`,
      fill: "url(#lossGrad)",
    });
    // 曲线
    add("path", { d: trainPath, fill: "none", stroke: "#5E6AD2", "stroke-width": "2.2", "stroke-linecap": "round", "stroke-linejoin": "round" });
    // 终点
    const endY = Y(trainLoss[trainLoss.length - 1]);
    add("circle", { cx: lastX, cy: endY, r: "4", fill: "#fff", stroke: "#5E6AD2", "stroke-width": "2" });
  }

  if (valLoss && valLoss.length > 1) {
    add("path", { d: path(valLoss), fill: "none", stroke: "#E07A3A", "stroke-width": "2.2", "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": "5,3" });
    // 终点
    const lastX = xOf(valLoss.length - 1, valLoss.length);
    const endY = Y(valLoss[valLoss.length - 1]);
    add("circle", { cx: lastX, cy: endY, r: "4", fill: "#fff", stroke: "#E07A3A", "stroke-width": "2" });
  }
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

async function renderLab(preselectedId = "") {
  clear(canvas);
  removeCompareBar();
  const completed = state.experiments.filter((e) => e.status === "completed");

  if (!completed.length) {
    renderLabEmpty();
    return;
  }

  const history = await loadAllLabHistory(completed);
  if (preselectedId === "new") {
    await renderLabNew(completed, completed[0].id, history);
    return;
  }
  const validPreselect = completed.some((e) => e.id === preselectedId) ? preselectedId : "";
  if (validPreselect || !history.length) {
    await renderLabNew(completed, validPreselect || completed[0].id, history);
    return;
  }
  renderLabHistoryHome(completed, history);
}

function labExperimentName(id) {
  const exp = state.experiments.find((e) => e.id === id);
  return exp ? exp.name : id;
}

async function loadAllLabHistory(completed) {
  const groups = await Promise.all(
    completed.map(async (exp) => {
      try {
        const data = await api(`/api/lab/history?experiment_id=${encodeURIComponent(exp.id)}`);
        return data.results || [];
      } catch {
        return [];
      }
    })
  );
  return groups.flat().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
}

function renderLabEmpty() {
  canvas.appendChild(
    el("div", { class: "view-head" }, [
      el("div", {}, [
        el("h1", { class: "view-title" }, "测评"),
        el("p", { class: "view-sub" }, "完成一次训练后，在这里对比训练前后的回答。"),
      ]),
    ])
  );
  canvas.appendChild(
    el("div", { class: "empty" }, [
      el("p", { class: "empty-title" }, "还没有可测评的训练结果"),
      el("p", {}, "先完成一次训练，再回来做单问对比或批量测评。"),
      el("button", { class: "btn primary", onClick: () => navigate("/new") }, "新建训练"),
    ])
  );
}

function renderLabHeader({ subtitle, actions = [] }) {
  canvas.appendChild(
    el("div", { class: "view-head lab-head" }, [
      el("div", {}, [
        el("h1", { class: "view-title" }, "测评"),
        el("p", { class: "view-sub" }, subtitle),
      ]),
      actions.length ? el("div", { class: "lab-head-actions" }, actions) : null,
    ])
  );
}

function renderLabHistoryHome(completed, history) {
  clear(canvas);
  renderLabHeader({
    subtitle: "回看每次训练前后的验证结果。",
    actions: [el("button", { class: "btn primary", onClick: () => renderLabNew(completed, completed[0].id, history) }, "新建测评")],
  });

  const experimentOptions = completed;
  if (
    state.labExperimentFilter !== "all" &&
    !experimentOptions.some((exp) => exp.id === state.labExperimentFilter)
  ) {
    state.labExperimentFilter = "all";
  }

  const toolbar = el("div", { class: "view-toolbar" }, [
    el("div", { class: "filters" }, [
      labFilterBtn("all", "全部", completed, history),
      labFilterBtn("compare", "单问对比", completed, history),
      labFilterBtn("batch", "批量测评", completed, history),
      labFilterBtn("chat", "自由对话", completed, history),
    ]),
    el("select", {
      class: "search",
      onChange: (e) => {
        state.labExperimentFilter = e.target.value;
        renderLabHistoryRows(list, completed, history);
      },
    }, [
      el("option", { value: "all", selected: state.labExperimentFilter === "all" }, "全部实验"),
    ].concat(
      experimentOptions.map((exp) =>
        el("option", { value: exp.id, selected: state.labExperimentFilter === exp.id }, exp.name)
      )
    )),
  ]);
  const list = el("div", { class: "lab-history-list" });

  canvas.appendChild(toolbar);
  canvas.appendChild(list);
  renderLabHistoryRows(list, completed, history);

  if (history.some((item) => labResultStatus(item) === "running")) {
    state.pollTimer = setTimeout(render, 2500);
  }
}

function labFilterBtn(key, label, completed, history) {
  return el("button", {
    class: `filter ${state.labFilter === key ? "active" : ""}`,
    onClick: () => {
      state.labFilter = key;
      renderLabHistoryHome(completed, history);
    },
  }, label);
}

function renderLabHistoryRows(list, completed, history) {
  clear(list);
  const filtered = filteredLabHistory(history);
  if (!filtered.length) {
    const selectedExperiment = completed.find((exp) => exp.id === state.labExperimentFilter);
    const hasTypeFilter = state.labFilter !== "all";
    const emptyTitle = selectedExperiment && !hasTypeFilter
      ? "这个实验还没有测评记录"
      : "没有匹配的测评记录";
    const emptyDesc = selectedExperiment && !hasTypeFilter
      ? "可以点右上角「新建测评」，先为这个实验跑一次验证。"
      : "换个类型或实验试试。";
    list.appendChild(
      el("div", { class: "empty" }, [
        el("p", { class: "empty-title" }, emptyTitle),
        el("p", {}, emptyDesc),
      ])
    );
    return;
  }
  filtered.forEach((item) => list.appendChild(labHistoryRow(item, completed, history)));
}

function filteredLabHistory(history) {
  let data = history;
  if (state.labFilter !== "all") {
    data = data.filter((item) => item.kind === state.labFilter);
  }
  if (state.labExperimentFilter !== "all") {
    data = data.filter((item) => item.experiment_id === state.labExperimentFilter);
  }
  return data;
}

function labHistoryRow(item, completed, history) {
  const questionCount = labResultItems(item).length || 1;
  const status = labResultStatus(item);
  const kindLabel = item.kind === "batch" ? "批量测评" : item.kind === "chat" ? "自由对话" : "单问对比";
  return el("div", { class: "lab-history-row" }, [
    el("button", { class: "lab-history-main", onClick: () => renderLabDetail(item, completed, history) }, [
      el("span", { class: "lab-history-title" }, item.kind === "batch" ? item.prompt || "批量测评" : item.kind === "chat" ? item.prompt || "自由对话" : item.prompt),
      el("span", { class: "lab-history-meta" }, [
        kindLabel,
        ` · ${labExperimentName(item.experiment_id)}`,
        item.kind === "chat" ? ` · ${(item.data?.messages || []).filter((m) => m.role === "user").length} 轮` : ` · ${questionCount} 条`,
        status === "running" ? " · 测评中" : "",
        status === "failed" ? " · 失败" : "",
        ` · ${formatDateTime(item.created_at)}`,
      ]),
    ]),
    el("button", {
      class: "history-delete",
      onClick: async () => {
        if (!confirm("删除这条测评记录？")) return;
        try {
          await api(`/api/lab/history/${item.id}`, { method: "DELETE" });
          await renderLab();
        } catch (err) {
          toast(err.message, true);
        }
      },
    }, "删除"),
  ]);
}

async function renderLabNew(completed, initialExperimentId, history = []) {
  clear(canvas);
  let activeExperimentId = initialExperimentId;
  let mode = "compare";
  let batchSource = "sample";

  const select = el(
    "select",
    { class: "lab-select" },
    completed.map((e) => el("option", { value: e.id, selected: activeExperimentId === e.id }, e.name))
  );
  const targetLine = el("div", { class: "lab-target" });
  const form = el("div", { class: "lab-new-form" });

  renderLabHeader({
    subtitle: history.length ? "选择训练结果，用一组问题验证训练前后的变化。" : "第一次测评：选择一个已完成的训练结果开始验证。",
    actions: history.length ? [
      el("button", { class: "btn", onClick: () => renderLabHistoryHome(completed, history) }, "返回历史"),
    ] : [],
  });

  canvas.appendChild(form);

  function paintTarget(message = "开始测评时会自动加载这个训练结果。") {
    clear(targetLine);
    targetLine.appendChild(el("span", { class: "muted" }, message));
  }

  async function loadSelectedExperiment() {
    const s = await api("/api/lab/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_id: activeExperimentId, use_adapter: true }),
    });
    paintTarget(`已加载「${s.experiment_name}」。`);
  }

  function paintForm() {
    clear(form);

    const modeChoices = el("div", { class: "choices lab-mode-choices" }, [
      el("button", {
        class: `choice ${mode === "compare" ? "active" : ""}`,
        onClick: () => { mode = "compare"; paintForm(); },
      }, [
        el("span", { class: "choice-title" }, "单问对比"),
        el("span", { class: "choice-note" }, "输入一个问题，立即看训练前后差异"),
      ]),
      el("button", {
        class: `choice ${mode === "batch" ? "active" : ""}`,
        onClick: () => { mode = "batch"; paintForm(); },
      }, [
        el("span", { class: "choice-title" }, "批量测评"),
        el("span", { class: "choice-note" }, "一次跑多条问题，适合看整体变化"),
      ]),
      el("button", {
        class: `choice ${mode === "chat" ? "active" : ""}`,
        onClick: () => { mode = "chat"; paintForm(); },
      }, [
        el("span", { class: "choice-title" }, "自由对话"),
        el("span", { class: "choice-note" }, "和训练后的模型持续聊天，感受效果"),
      ]),
    ]);

    form.appendChild(el("div", { class: "section-label" }, "1 · 选择训练结果"));
    form.appendChild(el("div", { class: "lab-experiment-picker" }, [select, targetLine]));
    form.appendChild(el("div", { class: "section-label" }, "2 · 选择测评方式"));
    form.appendChild(modeChoices);
    form.appendChild(el("div", { class: "section-label" }, "3 · 配置问题"));

    if (mode === "compare") {
      paintCompareForm();
    } else if (mode === "chat") {
      paintChatForm();
    } else {
      paintBatchForm();
    }
  }

  function paintStarters(promptHints, input, prompts) {
    clear(promptHints);
    if (!prompts.length) return;
    promptHints.appendChild(el("span", { class: "prompt-hint-label" }, "试试："));
    prompts.forEach((p) => {
      promptHints.appendChild(el("button", { class: "prompt-hint-btn", onClick: () => { input.value = p; } }, p));
    });
  }

  async function paintCompareForm() {
    const input = el("textarea", { class: "input", placeholder: "输入一个你最关心的问题…", rows: 3 });
    const promptHints = el("div", { class: "prompt-hints" });
    const resultBox = el("div", { class: "lab-run-result" });
    const startBtn = el("button", { class: "btn primary" }, "开始测评");

    form.appendChild(buildStyleControl());
    form.appendChild(input);
    form.appendChild(promptHints);
    form.appendChild(el("div", { class: "form-actions" }, [startBtn]));
    form.appendChild(resultBox);

    try {
      const starters = await api(`/api/lab/starters?experiment_id=${encodeURIComponent(activeExperimentId)}`);
      paintStarters(promptHints, input, starters.prompts || []);
    } catch {
      paintStarters(promptHints, input, []);
    }

    startBtn.addEventListener("click", async () => {
      const prompt = input.value.trim();
      if (!prompt) return toast("先输入一个问题。", true);
      startBtn.disabled = true;
      startBtn.textContent = "测评中…";
      clear(resultBox);
      resultBox.appendChild(el("p", { class: "muted" }, "正在生成训练前后回答，请稍候。"));
      try {
        await loadSelectedExperiment();
        const item = await api("/api/lab/compare-runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, style: state.labStyle, ...state.labAdvanced }),
        });
        renderLabDetail(item, completed, await loadAllLabHistory(completed));
      } catch (err) {
        clear(resultBox);
        resultBox.appendChild(el("p", { class: "text-error" }, `测评失败：${err.message}`));
      }
      startBtn.disabled = false;
      startBtn.textContent = "开始测评";
    });
  }

  function paintChatForm() {
    // 对话模式：选好风格后直接进入对话界面
    form.appendChild(buildStyleControl());
    const startBtn = el("button", { class: "btn primary" }, "开始对话");
    form.appendChild(el("div", { class: "form-actions" }, [startBtn]));

    startBtn.addEventListener("click", () => {
      renderLabChat(activeExperimentId, completed);
    });
  }

  function paintBatchForm() {
    const manualInput = el("textarea", { class: "input", placeholder: "每行一个问题，例如：\n你是谁？\n遇到退款问题怎么办？", rows: 5 });
    const sourceChoices = el("div", { class: "choices lab-mode-choices" }, [
      el("button", {
        class: `choice ${batchSource === "sample" ? "active" : ""}`,
        onClick: () => { batchSource = "sample"; paintForm(); },
      }, [
        el("span", { class: "choice-title" }, "从训练数据抽样"),
        el("span", { class: "choice-note" }, "默认推荐，系统自动选 5 条问题"),
      ]),
      el("button", {
        class: `choice ${batchSource === "manual" ? "active" : ""}`,
        onClick: () => { batchSource = "manual"; paintForm(); },
      }, [
        el("span", { class: "choice-title" }, "手动粘贴问题"),
        el("span", { class: "choice-note" }, "适合你已有明确验证问题"),
      ]),
    ]);
    const resultBox = el("div", { class: "lab-run-result" });
    const startBtn = el("button", { class: "btn primary" }, "开始批量测评");

    form.appendChild(sourceChoices);
    if (batchSource === "manual") form.appendChild(manualInput);
    else form.appendChild(el("p", { class: "batch-info" }, "系统会从当前训练数据中抽取问题，生成一条批量测评历史。"));
    form.appendChild(el("div", { class: "form-actions" }, [startBtn]));
    form.appendChild(resultBox);

    startBtn.addEventListener("click", async () => {
      const prompts = batchSource === "manual"
        ? manualInput.value.split("\n").map((x) => x.trim()).filter(Boolean)
        : [];
      if (batchSource === "manual" && !prompts.length) return toast("至少粘贴一个问题。", true);
      startBtn.disabled = true;
      startBtn.textContent = "批量测评中…";
      clear(resultBox);
      resultBox.appendChild(el("p", { class: "muted" }, "正在逐条生成对比结果，请稍候。"));
      try {
        await loadSelectedExperiment();
        const item = await api("/api/lab/batch-runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompts }),
        });
        renderLabDetail(item, completed, await loadAllLabHistory(completed));
      } catch (err) {
        clear(resultBox);
        resultBox.appendChild(el("p", { class: "text-error" }, `批量测评失败：${err.message}`));
      }
      startBtn.disabled = false;
      startBtn.textContent = "开始批量测评";
    });
  }

  select.addEventListener("change", () => {
    activeExperimentId = select.value;
    paintTarget();
    paintForm();
  });

  paintTarget();
  paintForm();
}

function renderLabDetail(item, completed, history) {
  clear(canvas);
  if (!item) {
    renderLabHistoryHome(completed, history);
    return;
  }

  // 对话类型使用专门的回看视图
  if (item.kind === "chat") {
    renderLabChatDetail(item, completed, history);
    return;
  }

  const rows = labResultItems(item);
  const typeLabel = item.kind === "batch" ? "批量测评" : "单问对比";
  const status = labResultStatus(item);
  renderLabHeader({
    subtitle: `${typeLabel} · ${labExperimentName(item.experiment_id)} · ${formatDateTime(item.created_at)}`,
    actions: [
      el("button", { class: "btn", onClick: () => renderLabHistoryHome(completed, history) }, "返回历史"),
      el("button", { class: "btn primary", onClick: () => renderLabNew(completed, item.experiment_id, history) }, "新建测评"),
    ],
  });

  canvas.appendChild(
    el("div", { class: "lab-detail-summary" }, [
      el("div", {}, [el("span", { class: "compare-label" }, "测评方式"), el("strong", {}, typeLabel)]),
      el("div", {}, [el("span", { class: "compare-label" }, "问题数"), el("strong", { class: "tnum" }, String(rows.length || 1))]),
      el("div", {}, [el("span", { class: "compare-label" }, "关联实验"), el("strong", {}, labExperimentName(item.experiment_id))]),
      el("div", {}, [el("span", { class: "compare-label" }, "状态"), el("strong", {}, labStatusLabel(status))]),
    ])
  );
  if (status === "failed") {
    canvas.appendChild(el("p", { class: "text-error lab-status-note" }, item.data?.error || "测评失败，请重新发起一次测评。"));
  }
  if (status === "running") {
    canvas.appendChild(el("p", { class: "muted lab-status-note" }, "测评正在后台生成。你可以切到其他页面，结果完成后会回到历史里。"));
    state.pollTimer = setTimeout(async () => {
      try {
        const fresh = await api(`/api/lab/history/${item.id}`);
        renderLabDetail(fresh, completed, await loadAllLabHistory(completed));
      } catch {
        render();
      }
    }, 2500);
  }
  canvas.appendChild(el("div", { class: "section-label" }, item.kind === "batch" ? "批量明细" : "训练前后对比"));
  canvas.appendChild(el("div", { class: "batch-results" }, rows.map((row, idx) => labResultCompareRow(row, idx, rows.length))));
}

function renderLabChatDetail(item, completed, history) {
  const chatMessages = item.data?.messages || [];
  renderLabHeader({
    subtitle: `自由对话 · ${labExperimentName(item.experiment_id)} · ${formatDateTime(item.created_at)}`,
    actions: [
      el("button", { class: "btn", onClick: () => renderLabHistoryHome(completed, history) }, "返回历史"),
    ],
  });

  canvas.appendChild(
    el("div", { class: "lab-detail-summary" }, [
      el("div", {}, [el("span", { class: "compare-label" }, "测评方式"), el("strong", {}, "自由对话")]),
      el("div", {}, [el("span", { class: "compare-label" }, "对话轮数"), el("strong", { class: "tnum" }, String(chatMessages.filter((m) => m.role === "user").length))]),
      el("div", {}, [el("span", { class: "compare-label" }, "关联实验"), el("strong", {}, labExperimentName(item.experiment_id))]),
    ])
  );

  // 渲染对话记录（只读）
  const messageList = el("div", { class: "chat-messages chat-messages-readonly" });
  for (const msg of chatMessages) {
    if (msg.role === "user") {
      messageList.appendChild(
        el("div", { class: "chat-msg chat-msg-user" }, [
          el("div", { class: "chat-bubble chat-bubble-user" }, msg.content),
        ])
      );
    } else if (msg.role === "assistant") {
      messageList.appendChild(
        el("div", { class: "chat-msg chat-msg-assistant" }, [
          el("div", { class: "chat-bubble chat-bubble-assistant" }, msg.content),
        ])
      );
    }
  }

  if (!chatMessages.length) {
    messageList.appendChild(el("div", { class: "empty" }, [
      el("p", { class: "empty-title" }, "这次对话没有消息记录"),
    ]));
  }

  canvas.appendChild(el("div", { class: "section-label" }, "对话记录"));
  canvas.appendChild(messageList);
}

function labResultStatus(item) {
  return item?.data?.status || "completed";
}

function labStatusLabel(status) {
  if (status === "running") return "测评中";
  if (status === "failed") return "失败";
  return "完成";
}

function labResultItems(item) {
  const savedRows = Array.isArray(item?.data?.results) ? item.data.results : [];
  if (savedRows.length) return savedRows;
  if (!item) return [];
  if (labResultStatus(item) === "running") {
    return [{
      prompt: item.prompt,
      base_answer: "生成中…",
      finetuned_answer: "生成中…",
    }];
  }
  return [{
    prompt: item.prompt,
    base_answer: item.base_answer,
    finetuned_answer: item.finetuned_answer,
  }];
}

function labResultCompareRow(item, idx, total) {
  return el("div", { class: "batch-row" }, [
    el("div", { class: "batch-question" }, total > 1 ? `${idx + 1}. ${item.prompt}` : item.prompt),
    el("div", { class: "compare-answers" }, [
      el("div", { class: "compare-col" }, [
        el("div", { class: "compare-label" }, "训练前"),
        el("div", { class: "compare-answer" }, item.base_answer || "无回答"),
      ]),
      el("div", { class: "compare-col" }, [
        el("div", { class: "compare-label accent" }, "训练后"),
        el("div", { class: "compare-answer" }, item.finetuned_answer || "无回答"),
      ]),
    ]),
  ]);
}

/* ---- 自由对话界面 ---- */
async function renderLabChat(experimentId, completed) {
  clear(canvas);
  removeCompareBar();

  const exp = completed.find((e) => e.id === experimentId);
  const expName = exp ? exp.name : "未命名实验";
  const messages = []; // 完整对话历史 [{role, content}]
  let sessionReady = false;

  // 顶部栏
  const header = el("div", { class: "chat-header" }, [
    el("div", { class: "chat-header-info" }, [
      el("span", { class: "chat-header-name" }, expName),
      el("span", { class: "chat-header-status", id: "chatStatus" }, "模型加载中…"),
    ]),
    el("div", { class: "chat-header-actions" }, [
      el("button", { class: "btn btn-sm", id: "chatRelease", onClick: handleRelease }, "释放模型"),
      el("button", { class: "btn btn-sm primary", id: "chatEnd", onClick: handleEnd }, "结束测评"),
    ]),
  ]);

  // 消息列表
  const messageList = el("div", { class: "chat-messages", id: "chatMessages" });

  // 输入区域
  const chatInput = el("textarea", {
    class: "chat-input",
    placeholder: "输入消息…",
    rows: 1,
    disabled: true,
  });
  const sendBtn = el("button", { class: "btn primary chat-send", disabled: true }, "发送");
  const inputBar = el("div", { class: "chat-input-bar" }, [chatInput, sendBtn]);

  // 组装
  const chatView = el("div", { class: "chat-view" }, [header, messageList, inputBar]);
  canvas.appendChild(chatView);

  // 自动调整输入框高度
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  // Enter 发送，Shift+Enter 换行
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  sendBtn.addEventListener("click", handleSend);

  // 加载提示
  appendSystemMessage("正在加载模型，请稍候…");

  // 启动 session
  try {
    await api("/api/lab/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_id: experimentId, use_adapter: true }),
    });
    sessionReady = true;
    document.getElementById("chatStatus").textContent = "模型已就绪";
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();
    clearSystemMessages();
    appendSystemMessage("模型已加载完成，开始聊吧。");
  } catch (err) {
    document.getElementById("chatStatus").textContent = "加载失败";
    appendSystemMessage(`模型加载失败：${err.message}`);
  }

  async function handleSend() {
    const text = chatInput.value.trim();
    if (!text || !sessionReady) return;

    // 添加用户消息
    messages.push({ role: "user", content: text });
    appendUserMessage(text);
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;
    chatInput.disabled = true;

    // 显示思考中
    const thinking = appendThinking();

    try {
      const result = await api("/api/lab/session/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, max_new_tokens: 200 }),
      });
      messages.push({ role: "assistant", content: result.content });
      thinking.remove();
      appendAssistantMessage(result.content, result.metrics);
    } catch (err) {
      thinking.remove();
      appendSystemMessage(`回复失败：${err.message}`);
    }

    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }

  async function handleEnd() {
    try {
      await api("/api/lab/session/end", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, experiment_id: experimentId }),
      });
    } catch {
      // 即使保存失败也允许退出
    }
    await renderLab();
  }

  async function handleRelease() {
    try {
      await api("/api/lab/session/end", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [], experiment_id: null }),
      });
    } catch {
      // ignore
    }
    sessionReady = false;
    document.getElementById("chatStatus").textContent = "已释放";
    chatInput.disabled = true;
    sendBtn.disabled = true;
    appendSystemMessage("模型已释放，对话结束。");
  }

  function appendUserMessage(text) {
    const msg = el("div", { class: "chat-msg chat-msg-user" }, [
      el("div", { class: "chat-bubble chat-bubble-user" }, text),
    ]);
    messageList.appendChild(msg);
    scrollToBottom();
  }

  function appendAssistantMessage(text, metrics) {
    const bubble = el("div", { class: "chat-bubble chat-bubble-assistant" }, text);
    const meta = metrics
      ? el("span", { class: "chat-msg-meta" }, `${metrics.generate_seconds}s · ${metrics.output_tokens} tokens`)
      : null;
    const msg = el("div", { class: "chat-msg chat-msg-assistant" }, meta ? [bubble, meta] : [bubble]);
    messageList.appendChild(msg);
    scrollToBottom();
  }

  function appendSystemMessage(text) {
    const msg = el("div", { class: "chat-msg chat-msg-system" }, [
      el("div", { class: "chat-bubble chat-bubble-system" }, text),
    ]);
    messageList.appendChild(msg);
    scrollToBottom();
  }

  function clearSystemMessages() {
    messageList.querySelectorAll(".chat-msg-system").forEach((el) => el.remove());
  }

  function appendThinking() {
    const msg = el("div", { class: "chat-msg chat-msg-assistant chat-thinking" }, [
      el("div", { class: "chat-bubble chat-bubble-assistant" }, [
        el("span", { class: "thinking-dots" }, "···"),
      ]),
    ]);
    messageList.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  function scrollToBottom() {
    messageList.scrollTop = messageList.scrollHeight;
  }
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
