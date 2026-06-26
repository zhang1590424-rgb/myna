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
    let payload = null;
    try {
      payload = await res.json();
      detail =
        typeof payload.detail === "string"
          ? payload.detail
          : payload.detail?.message || detail;
    } catch {
      detail = res.statusText || detail;
    }
    const err = new Error(detail);
    err.payload = payload;
    err.warnings = payload?.detail?.warnings || [];
    throw err;
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

function clear(node, { animate = true } = {}) {
  if (node === canvas) {
    node.classList.remove("chat-main");
    if (animate) {
      // trigger fade-in animation on view change
      node.classList.remove("fade-in");
      void node.offsetWidth; // force reflow
      node.classList.add("fade-in");
    }
  }
  node.replaceChildren();
}

function actionMenu(items) {
  const wrap = el("div", { class: "action-menu" });
  const btn = el("button", {
    class: "icon-btn action-menu-trigger",
    type: "button",
    title: "更多操作",
    "aria-label": "更多操作",
  }, "...");
  const panel = el("div", { class: "action-menu-panel" });
  for (const item of items) {
    const menuBtn = el("button", {
      class: `action-menu-item${item.danger ? " danger" : ""}`,
      type: "button",
      onClick: async (e) => {
        e.stopPropagation();
        wrap.classList.remove("open");
        await item.onClick(e);
      },
    }, item.label);
    panel.appendChild(menuBtn);
  }
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    document.querySelectorAll(".action-menu.open").forEach((node) => {
      if (node !== wrap) node.classList.remove("open");
    });
    wrap.classList.toggle("open");
  });
  wrap.addEventListener("click", (e) => e.stopPropagation());
  wrap.appendChild(btn);
  wrap.appendChild(panel);
  return wrap;
}

document.addEventListener("click", () => {
  document.querySelectorAll(".action-menu.open").forEach((node) => node.classList.remove("open"));
});

function showPreflightDialog(cards) {
  return new Promise((resolve) => {
    const overlay = el("div", { class: "preflight-overlay" });
    const dialog = el("div", { class: "preflight-dialog" });
    dialog.appendChild(el("div", { class: "preflight-title" }, "训练前数据检查"));
    dialog.appendChild(el("p", { class: "preflight-desc" }, "下面这些问题不会阻止训练，但会影响训练前后差异。"));

    const cardList = el("div", { class: "diag-cards" });
    for (const card of cards) {
      const levelClass = card.level === "error" ? "diag-error" : card.level === "warn" ? "diag-warn" : "diag-ok";
      const icon = "";
      const cardEl = el("div", { class: `diag-card ${levelClass}` }, [
        el("div", { class: "diag-card-header" }, [
          el("span", { class: "diag-icon" }, icon),
          el("span", { class: "diag-title" }, card.title),
        ]),
        el("div", { class: "diag-suggestion" }, card.suggestion),
      ]);
      cardList.appendChild(cardEl);
    }
    dialog.appendChild(cardList);

    const actions = el("div", { class: "preflight-actions" });
    const cancelBtn = el("button", { class: "btn ghost" }, "返回修改");
    const proceedBtn = el("button", { class: "btn primary" }, "继续训练");
    cancelBtn.addEventListener("click", () => { overlay.remove(); resolve(false); });
    proceedBtn.addEventListener("click", () => { overlay.remove(); resolve(true); });
    actions.appendChild(cancelBtn);
    actions.appendChild(proceedBtn);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
  });
}

function buildNumericStepper({ label, min, max, step, value, hint, changedText, onChange }) {
  const precision = step < 1 ? String(step).split(".")[1].length : 0;
  const clamp = (v) => {
    const clamped = Math.min(max, Math.max(min, v));
    return precision ? Number(clamped.toFixed(precision)) : clamped;
  };

  const range = el("input", {
    class: "param-range",
    type: "range",
    min,
    max,
    step,
    value,
  });
  const numInput = el("input", {
    class: "input tnum param-num",
    type: "number",
    min,
    max,
    step,
    value,
  });

  const sync = (v) => {
    const val = clamp(v);
    range.value = val;
    numInput.value = val;
    onChange(val);
  };

  range.addEventListener("input", () => sync(Number(range.value)));
  numInput.addEventListener("input", () => {
    const v = Number(numInput.value);
    if (!Number.isNaN(v)) {
      range.value = clamp(v);
      onChange(clamp(v));
    }
  });
  numInput.addEventListener("blur", () => { numInput.value = clamp(Number(numInput.value)); });

  const decBtn = el("button", { class: "param-step-btn", type: "button" }, "\u2212");
  const incBtn = el("button", { class: "param-step-btn", type: "button" }, "+");
  decBtn.addEventListener("click", () => sync(clamp(Number(numInput.value) - step)));
  incBtn.addEventListener("click", () => sync(clamp(Number(numInput.value) + step)));

  const stepper = el("div", { class: "param-stepper" }, [decBtn, numInput, incBtn]);

  const labelNode = el("label", {}, [
    label,
    changedText ? el("span", { class: "param-changed" }, changedText) : null,
  ]);

  return el("div", { class: "param-item" }, [labelNode, range, stepper, el("p", { class: "param-hint" }, hint)]);
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
let _metaPromise = null;
let _coreLoadedAt = 0;
const _CORE_STALE_MS = 5000; // 5 秒内视为新鲜，不重复拉取

async function refreshCore() {
  const [models, datasets, experiments] = await Promise.all([
    api("/api/models"),
    api("/api/datasets"),
    api("/api/experiments"),
  ]);
  state.models = models;
  state.datasets = datasets;
  state.experiments = experiments;
  _coreLoadedAt = Date.now();
}

/** 5 秒内不重复拉取 core 数据 */
async function refreshCoreIfStale() {
  if (Date.now() - _coreLoadedAt < _CORE_STALE_MS && state.experiments.length) return;
  await refreshCore();
}

async function refreshExperiments() {
  state.experiments = await api("/api/experiments");
}

async function ensureMeta() {
  if (state.presets.length && state.templates.length && state.environment) return;
  if (!_metaPromise) {
    _metaPromise = Promise.all([
      api("/api/training-presets"),
      api("/api/templates"),
      api("/api/environment"),
    ]).then(([presets, templates, environment]) => {
      state.presets = presets;
      state.templates = templates;
      state.environment = environment;
    });
  }
  await _metaPromise;
}

/** 启动时预取 meta，不阻塞首屏 */
function preloadMeta() {
  ensureMeta().catch(() => { _metaPromise = null; });
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
  const navView = ["new", "detail", "compare", "dataset-detail"].includes(view) ? (view === "dataset-detail" ? "datasets" : "experiments") : view;
  navButtons.forEach((b) => b.classList.toggle("active", b.dataset.view === navView));

  // 非首页 Tab 切换时立即显示 loading 占位，避免"点了没反应"
  if (view !== "home") {
    clear(canvas);
    canvas.appendChild(el("div", { class: "loading-placeholder" }, [
      el("span", { class: "loading-dot" }),
      el("span", {}, "加载中…"),
    ]));
  }

  try {
    if (view === "home") {
      await refreshExperiments();
      renderHome();
      refreshHomeLabHistory();
      maybePoll();
    } else if (view === "experiments") {
      await Promise.all([ensureMeta(), refreshCoreIfStale()]);
      renderExperiments();
      maybePoll();
    } else if (view === "new") {
      await Promise.all([ensureMeta(), refreshCoreIfStale()]);
      // 下载模型后环境状态可能变化，后台刷新环境（不阻塞渲染）
      api("/api/environment").then((env) => { state.environment = env; }).catch(() => {});
      renderNewExperiment(arg);
    } else if (view === "detail") {
      await Promise.all([ensureMeta(), refreshCoreIfStale()]);
      await renderDetail(arg);
    } else if (view === "compare") {
      await refreshCoreIfStale();
      await renderCompare(arg);
    } else if (view === "models") {
      state.models = await api("/api/models");
      renderModels();
    } else if (view === "datasets") {
      await Promise.all([
        api("/api/datasets").then((d) => { state.datasets = d; }),
        ensureMeta(),
      ]);
      if (arg === "new") {
        renderDatasetUpload();
      } else {
        renderDatasets();
      }
    } else if (view === "dataset-detail") {
      await renderDatasetDetail(arg);
    } else if (view === "lab") {
      await refreshCoreIfStale();
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
          renderExperiments({ animate: false });
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
function renderHome() {
  clear(canvas);
  removeCompareBar();

  if (!state.experiments.length) {
    renderHomeEmpty();
  } else {
    renderHomeDashboard();
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

function renderHomeDashboard(labHistory = []) {
  canvas.appendChild(renderHomeBanner({
    title: "Myna",
    subtitle: "训练一次，看见不同",
  }));

  canvas.appendChild(renderHomeShortcuts());
  canvas.appendChild(renderHomeActivityLists(latestExperiments(state.experiments, 6), labHistory.slice(0, 6)));
}

async function refreshHomeLabHistory() {
  const completed = state.experiments.filter((e) => e.status === "completed").slice(0, 10);
  if (!completed.length) return;
  let labHistory = [];
  try {
    const data = await api("/api/lab/history/recent?limit=6");
    labHistory = data.results || [];
  } catch {
    return;
  }
  // 只保留属于当前 completed 实验的测评记录，与测评页一致
  const completedIds = new Set(completed.map((e) => e.id));
  labHistory = labHistory.filter((item) => completedIds.has(item.experiment_id));
  if (parseHash().view !== "home") return;
  const panel = document.getElementById("homeEvaluationPanel");
  if (!panel) return;
  panel.replaceChildren(
    renderHomePanelHead("测评列表"),
    renderHomeEvaluationList(labHistory.slice(0, 6))
  );
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
    el("div", { class: "home-shortcuts stagger" },
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
    el("div", { class: "home-activity-panel", id: "homeEvaluationPanel" }, [
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

function renderExperiments({ animate = true } = {}) {
  clear(canvas, { animate });

  const head = el("div", { class: "view-head" }, [
    el("div", {}, [
      el("h1", { class: "view-title" }, "实验"),
      el(
        "p",
        { class: "view-sub" },
        `${state.experiments.length} 条训练记录`
      ),
    ]),
    el(
      "button",
      { class: "btn primary", onClick: () => navigate("/new") },
      "＋ 新建实验"
    ),
  ]);
  canvas.appendChild(head);

  const filtersDiv = el("div", { class: "filters" });
  function buildFilterBtns() {
    filtersDiv.replaceChildren();
    for (const [key, label] of [["all", "全部"], ["sft", "SFT"], ["dpo", "DPO"]]) {
      filtersDiv.appendChild(el("button", {
        class: `filter ${state.filter === key ? "active" : ""}`,
        onClick: () => {
          state.filter = key;
          buildFilterBtns();
          renderRows();
        },
      }, label));
    }
  }
  buildFilterBtns();

  const toolbar = el("div", { class: "view-toolbar" }, [
    filtersDiv,
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

  const table = el("div", { class: `exp-table${animate ? " stagger" : ""}` });
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

function experimentRow(exp) {
  const checked = state.selectedForCompare.has(exp.id);
  let row;
  const compareBtn = el("button", { class: `btn btn-sm compare-toggle${checked ? " active" : ""}` }, checked ? "已选" : "对比");
  compareBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.selectedForCompare.has(exp.id)) {
      state.selectedForCompare.delete(exp.id);
    } else {
      state.selectedForCompare.add(exp.id);
    }
    const nextChecked = state.selectedForCompare.has(exp.id);
    compareBtn.textContent = nextChecked ? "已选" : "对比";
    compareBtn.classList.toggle("active", nextChecked);
    row.classList.toggle("selected", nextChecked);
    renderCompareBar();
  });

  row = el(
    "div",
    {
      class: `exp-row ${checked ? "selected" : ""}`,
      onClick: () => navigate(`/detail/${exp.id}`),
    },
    [
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
      el("div", { class: "exp-actions" }, [
        compareBtn,
        actionMenu([
          {
            label: "删除实验",
            danger: true,
            onClick: async (e) => {
              await deleteExperiment(exp, async () => {
                state.experiments = await api("/api/experiments");
                state.selectedForCompare.delete(exp.id);
                renderExperiments();
              }, e.currentTarget);
            },
          },
        ]),
      ]),
    ]
  );
  return row;
}

async function deleteExperiment(exp, afterDelete, triggerBtn) {
  if (!confirm(`删除实验「${exp.name}」？训练产物和关联测评记录也会被移除。`)) return;
  if (triggerBtn) { triggerBtn.disabled = true; triggerBtn.textContent = "删除中…"; }
  try {
    await api(`/api/experiments/${exp.id}`, { method: "DELETE" });
    toast("已删除。");
    await afterDelete();
  } catch (err) {
    toast(err.message, true);
    if (triggerBtn) { triggerBtn.disabled = false; triggerBtn.textContent = "删除"; }
  }
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
  // 上传成功后跳转过来的 dataset id（只生效一次）
  const carriedDatasetId = state.pendingDatasetId || null;
  state.pendingDatasetId = null;
  const form = {
    model_id: source ? source.model_id : availableModels[0]?.id || null,
    dataset_id: source ? source.dataset_id : carriedDatasetId,
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
          "缺少训练组件或本地模型。先到模型页下载一个模型。"
        ),
      ])
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
  page.appendChild(fieldRow("训练模型", modelChoices));

  // ---- dataset field ----
  const datasetWrap = el("div", {});
  function compatibleDatasets() {
    const want = form.method === "dpo" ? "dpo_pairs" : "alpaca";
    return state.datasets.filter((d) => d.format === want);
  }
  function selectedDataset() {
    return state.datasets.find((d) => d.id === form.dataset_id) || null;
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
          paintPresetGuidance();
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
      ["sft", "SFT", "用问答样本教模型怎么回答"],
      ["dpo", "DPO", "用好/差回答教模型偏好"],
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
              paintPresetGuidance();
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
  const presetGuidance = el("p", { class: "muted" });
  function paintPresetGuidance() {
    const dataset = selectedDataset();
    if (!dataset) {
      presetGuidance.textContent = "先选数据，再决定训练强度。";
      return;
    }
    if (dataset.row_count < 30) {
      presetGuidance.textContent =
        `${dataset.row_count} 条数据偏少：先用「试跑」或「推荐」，训练后看回答变化。`;
      return;
    }
    if (dataset.row_count >= 200) {
      presetGuidance.textContent =
        `${dataset.row_count} 条数据较充足：先用「推荐」建立基线，需要对照时再用「加强」。`;
      return;
    }
    presetGuidance.textContent =
      `${dataset.row_count} 条数据：先用「推荐」，训练后去测评。`;
  }
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
    advanced.appendChild(el("summary", {}, "更多设置"));
    clear(paramGrid);
    const specs = [
      ["epochs", "训练轮数", 1, 30, 1, "同样的数据反复学几遍。越多学得越透，但太多会死记硬背。"],
      ["learning_rate", "学习率", 0.00001, 0.01, 0.00001, "每次调整的步子大小。太大容易学跑偏，太小学得慢。"],
      ["lora_rank", "LoRA rank", 1, 64, 1, "这次训练可写入的容量。越大越吃资源。"],
      ["batch_size", "批大小", 1, 16, 1, "一次同时看几条数据。越大越稳，但更占内存。"],
      ["grad_accum", "梯度累积步数", 1, 16, 1, "攒几批再更新一次模型。数据少就调小，模型学得更勤。"],
    ];
    if (form.method === "dpo") specs.push(["beta", "偏好强度 beta", 0.01, 1, 0.01, "多看重「更好/更差」的差距。越大越贴近你的偏好，太大易学偏。"]);
    for (const [key, label, min, max, step, hint] of specs) {
      const changed =
        sourceParams && Number(sourceParams[key]) !== Number(form.params[key]);
      paramGrid.appendChild(
        buildNumericStepper({
          label,
          min,
          max,
          step,
          value: form.params[key],
          hint,
          changedText: changed ? `（原 ${sourceParams[key]}）` : null,
          onChange: (v) => {
            form.presetId = null;
            form.params[key] = v;
            paintPresets();
          },
        })
      );
    }
    advanced.appendChild(paramGrid);

  }
  paintAdvanced();
  if (source) advanced.open = true;

  paramsWrap.appendChild(presetChoices);
  paramsWrap.appendChild(presetGuidance);
  paramsWrap.appendChild(advanced);
  paintPresetGuidance();
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
    if (!form.model_id) return toast("请先选一个可用模型。", true);
    if (!form.dataset_id) return toast("请先选一份匹配的训练数据。", true);
    startBtn.disabled = true;
    try {
      // Pre-flight data quality check
      const preflight = await api(`/api/datasets/${form.dataset_id}/preflight?method=${form.method}`);
      if (preflight.cards && preflight.cards.length > 0) {
        const proceed = await showPreflightDialog(preflight.cards);
        if (!proceed) {
          startBtn.disabled = false;
          return;
        }
      }
      const created = await api("/api/experiments", {
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
      navigate(`/experiments/${created.id}`);
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
async function renderDetail(id, { animate = true } = {}) {
  const exp = await api(`/api/experiments/${id}`);
  const route = parseHash();
  if (route.view !== "detail" || route.arg !== id) return;

  clear(canvas, { animate });
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
      el("button", { class: "btn primary", onClick: () => navigate(`/lab/${exp.id}`) }, "开始测评"),
      el("a", { class: "btn", href: `/api/experiments/${id}/export` }, "导出 LoRA"),
      el("a", { class: "btn", href: `/api/experiments/${id}/export?merge=true` }, "导出完整模型")
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
    actionMenu([
      {
        label: "删除实验",
        danger: true,
        onClick: async (e) => {
          await deleteExperiment(exp, async () => navigate("/experiments"), e.currentTarget);
        },
      },
    ])
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

  // Live tip — 训练中在图表上方显示单行提示（只描述现象，不给建议）
  const liveDiags = exp.status === "running" ? (exp.live_diagnostics || []) : [];
  let processTipNode = null;
  if (liveDiags.length) {
    // 按严重程度取最重要一条：error > warn > ok
    const priority = { error: 3, warn: 2, ok: 1 };
    const top = liveDiags.reduce((a, b) => (priority[b.level] || 0) > (priority[a.level] || 0) ? b : a);
    const liveText = top.suggestion ? `${top.title}，${top.suggestion}` : top.title;
    processTipNode = el("div", { class: `process-tip process-tip-${top.level}` }, [
      el("span", { class: "process-tip-label" }, "过程提示"),
      el("span", { class: "process-tip-text" }, liveText),
    ]);
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
    if (processTipNode) effect.appendChild(processTipNode);

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
    const metricsRow = el("div", { class: "loss-metrics-row" }, [
      el("div", { class: "loss-metric-item" }, [
        el("div", { class: "loss-metric-label" }, "初始 Loss"),
        el("div", { class: "loss-metric-val" }, fmtLoss(startLoss)),
      ]),
      el("div", { class: "loss-metric-item positive" }, [
        el("div", { class: "loss-metric-label" }, "当前 Loss"),
        el("div", { class: "loss-metric-val" }, fmtLoss(currentLoss)),
      ]),
      el("div", { class: "loss-metric-item" }, [
        el("div", { class: "loss-metric-label" }, "总步数"),
        el("div", { class: "loss-metric-val" }, String(exp.loss.length)),
      ]),
      hasEval
        ? el("div", { class: "loss-metric-item warn" }, [
            el("div", { class: "loss-metric-label" }, "验证最低"),
            el("div", { class: "loss-metric-val" }, fmtLoss(minEvalLoss)),
          ])
        : el("div", { class: "loss-metric-item" }, [
            el("div", { class: "loss-metric-label" }, "下降幅度"),
            el("div", { class: "loss-metric-val" }, dropPct !== "—" ? `${dropPct}%` : "—"),
          ]),
    ]);

    const body = el("div", { class: "loss-body" }, [chartCol, metricsRow]);
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

  // 训练完成后的结果解读模块（体检报告式：结论 → 按主题逐行检查 → 展开详情）
  const finalDiags = exp.status === "completed" ? (exp.diagnostics || []) : [];
  if (finalDiags.length) {
    const diagSection = el("section", { class: "detail-section result-insights" });
    diagSection.appendChild(el("div", { class: "section-label" }, "结果解读"));

    const topicMeta = {
      train_loss: { title: "训练曲线" },
      train_process: { title: "训练过程" },
      eval_loss: { title: "泛化判断" },
      train_eval: { title: "训练与验证" },
      data: { title: "数据质量" },
      dpo: { title: "偏好学习" },
      general: { title: "其他检查" },
    };
    const referenceSignals = [
      "没有验证", "没有拿到可解读", "波动较大",
      "比训练 loss 表现还好", "数据切分", "参考",
    ];
    const getCardStatus = (card) => {
      if (card.level === "error") return "error";
      if (card.level === "warn") return "warn";
      const text = [card.title, card.observation, card.interpretation, card.suggestion]
        .filter(Boolean).join(" ");
      if (referenceSignals.some((s) => text.includes(s))) return "reference";
      return "ok";
    };

    // Same-topic de-dup: if topic has warn/error, drop its ok cards
    const topicsWithIssue = new Set(
      finalDiags
        .filter((c) => c.level === "warn" || c.level === "error")
        .map((c) => c.topic || "general")
    );
    const cards = finalDiags.filter(
      (c) => !(c.level === "ok" && topicsWithIssue.has(c.topic || "general"))
    );

    // Group by topic
    const topicGroups = {};
    for (const card of cards) {
      const topic = card.topic || "general";
      if (!topicGroups[topic]) topicGroups[topic] = [];
      topicGroups[topic].push(card);
    }

    // Determine worst status per topic
    const statusRank = { error: 0, warn: 1, reference: 2, ok: 3 };
    const topicEntries = Object.entries(topicGroups).map(([topic, topicCards]) => {
      let worstStatus = "ok";
      for (const c of topicCards) {
        const s = getCardStatus(c);
        if (statusRank[s] < statusRank[worstStatus]) worstStatus = s;
      }
      return { topic, cards: topicCards, worstStatus };
    });
    topicEntries.sort((a, b) => statusRank[a.worstStatus] - statusRank[b.worstStatus]);

    // Overall conclusion
    const hasError = topicEntries.some((e) => e.worstStatus === "error");
    const hasWarn = topicEntries.some((e) => e.worstStatus === "warn");
    const issueCount = topicEntries.filter((e) => e.worstStatus === "error" || e.worstStatus === "warn").length;
    const leadCard = topicEntries[0]?.cards[0];

    const summaryTitle = hasError
      ? "先处理训练异常"
      : hasWarn
        ? "先测评，再按风险调整"
        : "训练过程正常，进入测评";
    const summaryText = hasError
      ? `最优先处理：${leadCard?.title || "训练异常"}。处理后再重新训练或测评。`
      : hasWarn
        ? `有 ${issueCount} 个风险信号会影响效果判断。先用训练集外问题测评，如果效果不明显，再按风险项调整。`
        : "训练过程没有明显失败信号。下一步用真实问题对比训练前后的回答变化。";

    const primaryAction = hasError
      ? { label: "检查数据", action: "goto_data" }
      : { label: "去测评", action: "goto_eval" };

    // Build report
    const report = el("div", { class: "dx-report" });

    // Conclusion
    const conclusionBtn = el("button", { class: "dx-conclusion-action" }, primaryAction.label);
    conclusionBtn.addEventListener("click", () => {
      if (primaryAction.action === "goto_eval") navigate(`/lab/${exp.id}`);
      else if (primaryAction.action === "goto_data") navigate("/datasets");
    });
    report.appendChild(el("div", { class: "dx-conclusion" }, [
      el("div", { class: "dx-conclusion-body" }, [
        el("h2", { class: "dx-conclusion-title" }, summaryTitle),
        summaryText ? el("p", { class: "dx-conclusion-text" }, summaryText) : null,
      ]),
      conclusionBtn,
    ]));

    // Checklist
    const statusLabels = { ok: "正常", warn: "注意", error: "异常", reference: "参考" };
    const chevronSvg = '<svg class="dx-chevron" viewBox="0 0 12 12" fill="none"><path d="M4.5 2.5L8 6L4.5 9.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    const checklist = el("div", { class: "dx-checklist" });

    for (const entry of topicEntries) {
      const meta = topicMeta[entry.topic] || topicMeta.general;
      const row = el("div", { class: "dx-row" });

      // Header
      const dotMod = entry.worstStatus === "ok" ? ""
        : entry.worstStatus === "error" ? " dx-row-dot-error"
        : entry.worstStatus === "warn" ? " dx-row-dot-warn"
        : " dx-row-dot-reference";
      const header = el("div", { class: "dx-row-header" });
      header.innerHTML = [
        `<span class="dx-row-dot${dotMod}"></span>`,
        `<span class="dx-row-info"><span class="dx-row-topic">${meta.title}</span><span class="dx-row-title">${entry.cards[0].title || ""}</span></span>`,
        `<span class="dx-row-status dx-row-status-${entry.worstStatus}">${statusLabels[entry.worstStatus]}${chevronSvg}</span>`,
      ].join("");
      header.addEventListener("click", () => row.classList.toggle("dx-open"));
      row.appendChild(header);

      // Expanded findings
      const expand = el("div", { class: "dx-expand" });
      for (const card of entry.cards) {
        const finding = el("div", { class: "dx-finding" });
        finding.appendChild(el("h4", { class: "dx-finding-title" }, card.title));
        if (card.observation) {
          finding.appendChild(el("p", { class: "dx-finding-observation" }, card.observation));
        }
        finding.appendChild(el("p", { class: "dx-finding-next" }, card.next_step || card.suggestion || "先看训练前后的测评结果。"));

        // Detail fold
        const parts = [];
        if (card.interpretation) parts.push(["怎么理解", card.interpretation]);
        if (card.mechanism) parts.push(["背后原因", card.mechanism]);
        if (card.how_to_tell) parts.push(["怎么分辨", card.how_to_tell]);
        if (card.evidence) parts.push(["依据", card.evidence]);
        if (parts.length) {
          const details = el("details", { class: "dx-finding-details" });
          details.appendChild(el("summary", {}, "判断依据"));
          const body = el("div", { class: "dx-finding-details-body" });
          for (const [label, text] of parts) {
            body.appendChild(el("p", {}, [el("strong", {}, `${label}：`), text]));
          }
          details.appendChild(body);
          finding.appendChild(details);
        }

        // Per-finding action (hide if same as primary)
        if (card.action && card.action.action !== primaryAction.action) {
          const actDiv = el("div", { class: "dx-finding-action" });
          const actBtn = el("button", {}, card.action.label);
          actBtn.addEventListener("click", () => {
            if (card.action.action === "goto_eval") navigate(`/lab/${exp.id}`);
            else if (card.action.action === "goto_data") navigate("/datasets");
            else if (card.action.action === "retrain") navigate(`/new/${exp.id}`);
          });
          actDiv.appendChild(actBtn);
          finding.appendChild(actDiv);
        }
        expand.appendChild(finding);
      }
      row.appendChild(expand);
      checklist.appendChild(row);
    }

    report.appendChild(checklist);
    diagSection.appendChild(report);
    canvas.appendChild(diagSection);
  }

  // notes
  canvas.appendChild(el("div", { class: "section-label" }, "实验笔记"));
  const notes = el("textarea", {
    class: "input notes-area",
    placeholder: "跑完随手记：这次改了什么、效果如何、下次试什么…",
  });
  notes.value = exp.notes || "";
  const saveNotes = el("button", { class: "btn btn-sm", style: "margin-top:10px" }, "保存笔记");
  saveNotes.addEventListener("click", async () => {
    saveNotes.disabled = true;
    saveNotes.textContent = "保存中…";
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
    saveNotes.disabled = false;
    saveNotes.textContent = "保存笔记";
  });
  canvas.appendChild(notes);
  canvas.appendChild(el("div", {}, [saveNotes]));

  // poll if live
  if (["running", "queued", "stopping", "pending"].includes(exp.status)) {
    state.pollTimer = setTimeout(async () => {
      const route = parseHash();
      if (route.view !== "detail" || route.arg !== id) return;
      try {
        await renderDetail(id, { animate: false });
      } catch {
        // Keep the current detail page visible if a transient refresh fails.
      }
    }, 2500);
  }
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
    add("circle", { cx: lastX, cy: endY, r: "4", fill: "#FEFEFC", stroke: "#5E6AD2", "stroke-width": "2" });
  }

  if (valLoss && valLoss.length > 1) {
    add("path", { d: path(valLoss), fill: "none", stroke: "#E07A3A", "stroke-width": "2.2", "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": "5,3" });
    // 终点
    const lastX = xOf(valLoss.length - 1, valLoss.length);
    const endY = Y(valLoss[valLoss.length - 1]);
    add("circle", { cx: lastX, cy: endY, r: "4", fill: "#FEFEFC", stroke: "#E07A3A", "stroke-width": "2" });
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
        el("p", { class: "view-sub" }, "下载到本机后才能训练。"),
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
        `${state.datasets.length} 份数据集`
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

/* ---- 上传数据：与"新建训练"对齐的草稿确认流 ---- *
 * 上传后数据进入"草稿态"：原地展示文件元信息 + 诊断（纯文本流）+ 预览。
 * 底部固定「取消 / 使用这份数据」操作栏：取消 = DELETE 已落库的草稿；
 * 使用这份数据 = 跳详情；error 等级时按钮置灰，强制用户先换一份。
 */
function renderDatasetUpload() {
  clear(canvas);
  removeCompareBar();

  const view = el("div", { class: "form-view dataset-upload" });
  canvas.appendChild(view);
  view.appendChild(
    el("button", { class: "back-link", onClick: () => navigate("/datasets") }, "← 返回数据列表")
  );
  view.appendChild(el("h1", { class: "view-title" }, "上传数据"));

  const page = el("div", { class: "form-page" });
  view.appendChild(page);

  // 上传前：选类型 + dropzone；上传后：dropzone 折成一行 + 下方长出诊断/预览
  const formatOptions = [
    { value: "alpaca", title: "问答数据（SFT）", note: "一问一答的对话格式" },
    { value: "dpo_pairs", title: "偏好数据（DPO）", note: "同一个问题的好回答和差回答" },
  ];
  let selectedFormat = formatOptions[0].value;
  let currentDatasetId = null;       // 上传成功后记录的 dataset_id
  let currentTopLevel = "ok";        // 诊断等级，影响「使用这份数据」是否可点

  // === 数据类型选择区 ===
  const typeSection = el("div", { class: "upload-step" });
  typeSection.appendChild(el("div", { class: "section-label" }, "数据类型"));
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
        if (currentDatasetId) return;  // 已上传则禁止改类型
        selectedFormat = opt.value;
        paintTypeChoices();
      });
      typeChoices.appendChild(card);
    }
  }
  paintTypeChoices();
  typeSection.appendChild(typeChoices);
  page.appendChild(typeSection);

  // === 上传区 ===
  const uploadSection = el("div", { class: "upload-step" });
  uploadSection.appendChild(el("div", { class: "section-label" }, "上传文件"));
  const fileInput = el("input", { type: "file", accept: ".csv,.json,.jsonl,.xlsx", hidden: true });
  const drop = el("div", { class: "dropzone" });

  function paintDropIdle() {
    clear(drop);
    drop.classList.remove("uploading", "drag", "is-replace");
    drop.appendChild(el("span", {}, [
      "把文件拖到这里，或 ",
      el("span", { class: "pick", onClick: () => fileInput.click() }, "点击选择"),
    ]));
    drop.appendChild(el("span", { class: "dropzone-hint" }, "支持 CSV · JSON · JSONL · XLSX"));
    drop.appendChild(fileInput);
  }
  function paintDropUploading(text = "上传中，请稍候…") {
    clear(drop);
    drop.classList.add("uploading");
    drop.appendChild(el("span", { class: "muted" }, text));
    drop.appendChild(fileInput);
  }
  function paintDropDone(result, formatLabel) {
    clear(drop);
    drop.classList.remove("uploading", "drag");
    drop.classList.add("is-replace");
    drop.appendChild(
      el("div", { class: "dropzone-done" }, [
        el("div", { class: "dropzone-done-info" }, [
          el("span", { class: "dropzone-done-filename" }, result.filename),
          el(
            "span",
            { class: "dropzone-done-meta" },
            `${formatLabel} · ${result.valid_count} 条`
          ),
        ]),
        el(
          "button",
          { class: "btn btn-sm ghost", type: "button", onClick: () => fileInput.click() },
          "换一份文件"
        ),
      ])
    );
    drop.appendChild(fileInput);
  }
  paintDropIdle();
  uploadSection.appendChild(drop);
  page.appendChild(uploadSection);

  // 示例数据：仅未上传时显示
  const sampleSection = el("div", { class: "upload-step upload-samples" });
  if (state.templates.length) {
    sampleSection.appendChild(el("div", { class: "section-label" }, "没有数据？下载示例先跑通"));
    sampleSection.appendChild(
      el(
        "div",
        { class: "sample-links" },
        state.templates.map((t) =>
          el("a", { class: "btn btn-sm", href: `/api/sample-data/${t.id}` }, t.title)
        )
      )
    );
    page.appendChild(sampleSection);
  }

  // === 结果区（诊断 + 预览，原地长出）===
  const resultMount = el("div", { class: "upload-result hidden" });
  page.appendChild(resultMount);

  // === 底部固定操作栏（与新建训练对齐）===
  const cancelBtn = el(
    "button",
    {
      class: "btn ghost",
      type: "button",
      onClick: async () => {
        if (currentDatasetId) {
          // 已落库的草稿数据：先 DELETE，再回上传初始态
          cancelBtn.disabled = true;
          try {
            await api(`/api/datasets/${currentDatasetId}`, { method: "DELETE" });
            state.datasets = await api("/api/datasets");
          } catch {
            // 删失败不影响离开
          }
        }
        navigate("/datasets");
      },
    },
    "取消"
  );
  const confirmBtn = el(
    "button",
    {
      class: "btn primary",
      type: "button",
      onClick: () => {
        if (!currentDatasetId || currentTopLevel === "error") return;
        toast("数据已创建。");
        navigate(`/dataset-detail/${currentDatasetId}`);
      },
    },
    "使用这份数据"
  );
  page.appendChild(el("div", { class: "form-actions" }, [cancelBtn, confirmBtn]));

  function refreshConfirmState() {
    if (!currentDatasetId) {
      confirmBtn.disabled = true;
      confirmBtn.title = "上传文件后才能创建";
    } else if (currentTopLevel === "error") {
      confirmBtn.disabled = true;
      confirmBtn.title = "数据存在需要先处理的问题，请换一份合规的数据";
    } else {
      confirmBtn.disabled = false;
      confirmBtn.title = "";
    }
  }
  refreshConfirmState();

  async function upload(file) {
    if (!file) return;
    paintDropUploading(currentDatasetId ? "更新中，请稍候…" : "上传中，请稍候…");
    clear(resultMount);
    resultMount.classList.add("hidden");
    const fd = new FormData();
    fd.append("file", file);
    if (!currentDatasetId) fd.append("format", selectedFormat);
    try {
      const path = currentDatasetId
        ? `/api/datasets/${currentDatasetId}`
        : "/api/datasets";
      const method = currentDatasetId ? "PUT" : "POST";
      const result = await api(path, { method, body: fd });
      state.datasets = await api("/api/datasets");
      currentDatasetId = result.dataset_id;
      const flagged = [
        ...(result.diagnostics || []).filter((c) => c.level !== "ok"),
        ...((result.warnings || []).map(() => ({ level: "warn" }))),
      ];
      currentTopLevel = flagged.some((c) => c.level === "error")
        ? "error"
        : flagged.length
        ? "warn"
        : "ok";
      const formatLabel = result.format === "dpo_pairs" ? "偏好 DPO" : "问答 SFT";
      paintDropDone(result, formatLabel);
      sampleSection.classList.add("hidden");
      renderUploadResultInto(resultMount, result);
      refreshConfirmState();
    } catch (err) {
      currentTopLevel = "error";
      renderUploadErrorInto(resultMount, err, Boolean(currentDatasetId));
      if (currentDatasetId) {
        paintDropDone(
          { filename: "（已保存的旧文件）", valid_count: "—" },
          selectedFormat === "dpo_pairs" ? "偏好 DPO" : "问答 SFT"
        );
      } else {
        paintDropIdle();
      }
      refreshConfirmState();
    } finally {
      fileInput.value = "";
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
}

/* 把"上传结果"渲染进给定容器：数据检查 + 预览。
 * 不再有状态条容器、不再有行内按钮，主操作集中到页面底部的「使用这份数据」。
 */
function renderUploadResultInto(mount, result) {
  clear(mount);
  mount.classList.remove("hidden");

  const diagnostics = Array.isArray(result.diagnostics) ? result.diagnostics : [];
  const warningCards = (result.warnings || []).map((w) => buildUploadWarningCard(w));
  const allCards = [...diagnostics, ...warningCards];
  const flagged = allCards.filter((c) => c.level !== "ok");
  const topLevel = flagged.some((c) => c.level === "error")
    ? "error"
    : flagged.length
    ? "warn"
    : "ok";

  // 预览表（先建好，便于诊断行号 chip 联动）
  const previewRows = Array.isArray(result.preview) ? result.preview : [];
  const previewTable = buildPreviewTable(previewRows);

  if (flagged.length) {
    appendUploadCheck(
      mount,
      renderDiagFlow(allCards, flagged, topLevel, result, previewTable)
    );
  }

  // 数据预览
  if (previewTable) {
    const previewBlock = el("div", { class: "upload-preview" });
    previewBlock.appendChild(
      el("div", { class: "section-label" }, `数据预览（前 ${previewRows.length} 条）`)
    );
    previewBlock.appendChild(previewTable);
    mount.appendChild(previewBlock);
  }
}

function renderUploadErrorInto(mount, err, hasSavedDraft) {
  clear(mount);
  mount.classList.remove("hidden");

  const warnings = Array.isArray(err?.warnings) ? err.warnings : [];
  const cards = [
    {
      level: "error",
      title: err?.message || "文件没有通过数据检查。",
      suggestion: hasSavedDraft
        ? "新文件没有保存，下面不会继续展示旧文件的检查结果。请按提示修改后重新上传。"
        : "请按提示修改文件后重新上传。",
      topic: "data",
    },
    ...warnings.map((w) => buildUploadWarningCard(w)),
  ];
  const flagged = cards.filter((c) => c.level !== "ok");
  const summary = `数据检查未通过，有 ${flagged.length} 个问题需要先处理。`;

  appendUploadCheck(
    mount,
    renderDiagFlow(
      cards,
      flagged,
      "error",
      { check_summary: summary },
      null
    )
  );
}

function appendUploadCheck(mount, flow) {
  mount.appendChild(
    el("div", { class: "upload-check" }, [
      el("div", { class: "section-label" }, "数据检查"),
      flow,
    ])
  );
}

function buildUploadWarningCard(text) {
  return {
    level: "warn",
    title: text,
    suggestion: "这一行不会进入训练，请补齐后重新上传。",
    topic: "data",
  };
}

/* 诊断流：一句话总结 + 诊断卡片列表。
 * 卡片样式对齐全局 preflight 弹窗的 .diag-card（圆点 + 标题 + 柔和色块 + 副文本），
 * 额外挂上行号联动与「为什么」折叠。
 */
function renderDiagFlow(allCards, flagged, topLevel, result, previewTable) {
  const wrap = el("div", { class: `diag-flow diag-flow-${topLevel}` });
  wrap.appendChild(
    el("div", { class: "diag-flow-summary" }, buildDiagSummary(topLevel, result, flagged))
  );
  if (flagged.length) {
    const cards = el("div", { class: "diag-cards" });
    for (const card of flagged) cards.appendChild(buildDiagCard(card, previewTable));
    wrap.appendChild(cards);
  }
  return wrap;
}

function buildDiagSummary(level, result, flagged) {
  if (result?.check_summary) return result.check_summary;
  const count = result.valid_count;
  if (level === "ok") return `数据可以创建，共 ${count} 条数据，没有发现明显问题。`;
  if (level === "warn") return `数据可以创建，共 ${count} 条数据，有 ${flagged.length} 个地方建议优化。`;
  const errCount = flagged.filter((c) => c.level === "error").length;
  return `数据检查未通过，共 ${count} 条数据，有 ${errCount} 个问题需要先处理。`;
}

/* 单张诊断卡片：复用全局 .diag-card 视觉，圆点表示等级。 */
function buildDiagCard(card, previewTable) {
  const level = card.level || "warn";
  const levelClass = level === "error" ? "diag-error" : level === "warn" ? "diag-warn" : "diag-ok";
  const cardEl = el("div", { class: `diag-card ${levelClass}` });

  cardEl.appendChild(
    el("div", { class: "diag-card-header" }, [
      el("span", { class: "diag-icon" }),
      el("span", { class: "diag-title" }, card.title),
    ])
  );

  if (card.suggestion) {
    cardEl.appendChild(el("div", { class: "diag-suggestion" }, card.suggestion));
  }

  // 元信息行：行号 + 「为什么」，统一缩进对齐标题文字
  const lineNumbers = card.evidence ? extractLineNumbers(card.evidence) : [];
  const why = card.interpretation || card.mechanism;
  if (lineNumbers.length || why) {
    const meta = el("div", { class: "diag-card-meta" });

    if (lineNumbers.length) {
      const lines = el("span", { class: "diag-lines" });
      lines.appendChild(el("span", { class: "diag-lines-label" }, "行号"));
      const shown = lineNumbers.slice(0, 10);
      for (const n of shown) {
        if (previewTable) {
          const chip = el("button", { class: "diag-line-chip", type: "button" }, String(n));
          chip.addEventListener("click", () => highlightPreviewRow(previewTable, n));
          lines.appendChild(chip);
        } else {
          lines.appendChild(el("span", { class: "diag-line-chip is-static" }, String(n)));
        }
      }
      if (lineNumbers.length > shown.length) {
        lines.appendChild(el("span", { class: "diag-line-more" }, `+${lineNumbers.length - shown.length}`));
      }
      meta.appendChild(lines);
    }

    if (why) {
      const toggle = el("button", { class: "diag-why-toggle", type: "button" }, "为什么");
      const detail = el("div", { class: "diag-why hidden" }, why);
      toggle.addEventListener("click", () => {
        const open = detail.classList.toggle("hidden");
        toggle.textContent = open ? "为什么" : "收起";
      });
      meta.appendChild(toggle);
      cardEl.appendChild(meta);
      cardEl.appendChild(detail);
    } else {
      cardEl.appendChild(meta);
    }
  }

  return cardEl;
}

/* ---------- 诊断流（详情页用，无 result.valid_count 上下文） ---------- */
function renderDiagFlowForDetail(cards, previewTable) {
  const flagged = cards.filter((c) => c.level !== "ok");
  if (!flagged.length) {
    return el("div", { class: "diag-flow diag-flow-ok" }, [
      el("div", { class: "diag-flow-summary" }, "数据体检通过，没有发现问题。"),
    ]);
  }
  const topLevel = flagged.some((c) => c.level === "error") ? "error" : "warn";
  const summary =
    topLevel === "error"
      ? `发现 ${flagged.filter((c) => c.level === "error").length} 处需要先处理：`
      : `发现 ${flagged.length} 处可以优化的地方：`;
  const wrap = el("div", { class: `diag-flow diag-flow-${topLevel}` });
  wrap.appendChild(el("div", { class: "diag-flow-summary" }, summary));
  const cardList = el("div", { class: "diag-cards" });
  for (const card of flagged) cardList.appendChild(buildDiagCard(card, previewTable));
  wrap.appendChild(cardList);
  return wrap;
}

function extractLineNumbers(text) {
  if (!text) return [];
  // 匹配「第 5、12、18 行」「涉及第 3 行」等
  const nums = [];
  const re = /\d+/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const n = parseInt(m[0], 10);
    if (!Number.isNaN(n) && n > 0 && n < 100000) nums.push(n);
  }
  // 去重并保留顺序
  return [...new Set(nums)];
}

function buildPreviewTable(rows) {
  if (!rows || !rows.length) return null;
  const cols = Object.keys(rows[0]);
  // 表头是第 1 行，数据从第 2 行起。每行 data-line 用于诊断 chip 跳转。
  const tbody = el("tbody", {});
  rows.forEach((r, idx) => {
    const lineNum = idx + 2;
    const tr = el(
      "tr",
      { "data-line": String(lineNum) },
      [
        el("td", { class: "preview-line-col" }, String(lineNum)),
        ...cols.map((c) => el("td", {}, String(r[c] ?? ""))),
      ]
    );
    tbody.appendChild(tr);
  });
  return el("table", { class: "preview-table preview-table-numbered" }, [
    el("thead", {}, [
      el("tr", {}, [
        el("th", { class: "preview-line-col" }, "行"),
        ...cols.map((c) => el("th", {}, c)),
      ]),
    ]),
    tbody,
  ]);
}

function highlightPreviewRow(table, lineNumber) {
  if (!table) return;
  const target = table.querySelector(`tr[data-line="${lineNumber}"]`);
  if (!target) return;
  // 清除旧高亮
  for (const tr of table.querySelectorAll("tr.is-highlight")) {
    tr.classList.remove("is-highlight");
  }
  target.classList.add("is-highlight");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
}

function datasetHealth(d) {
  const min = d.format === "dpo_pairs" ? 20 : 30;
  const hardMin = d.format === "dpo_pairs" ? 10 : 10;
  if (d.row_count < hardMin) {
    return { label: "样本太少", className: "danger", text: "建议先补数据" };
  }
  if (d.row_count < min) {
    return { label: "可试跑", className: "warn", text: "适合跑通流程" };
  }
  return { label: "可训练", className: "ok", text: "适合新建实验" };
}

function datasetRow(d) {
  const health = datasetHealth(d);
  return el("div", { class: "list-row clickable", onClick: () => navigate(`/dataset-detail/${d.id}`) }, [
    el("div", {}, [
      el("div", { class: "row-name" }, [
        el("span", { class: "row-name-text", title: d.name }, d.name),
        el("span", { class: "badge-ok" }, d.format === "dpo_pairs" ? "偏好 DPO" : "问答 SFT"),
        el("span", { class: `data-health data-health-${health.className}` }, health.label),
      ]),
      el(
        "div",
        { class: "row-note" },
        `${d.row_count} 条 · ${health.text} · 来自 ${d.source_filename} · ${relativeTime(d.created_at)}`
      ),
    ]),
    el("div", { class: "row-actions" }, [
      actionMenu([
        {
          label: "删除数据",
          danger: true,
          onClick: async (e) => {
            if (!confirm(`删除数据集「${d.name}」？`)) return;
            const delBtn = e.currentTarget;
            delBtn.disabled = true; delBtn.textContent = "删除中…";
            try {
              await api(`/api/datasets/${d.id}`, { method: "DELETE" });
              toast("已删除。");
              state.datasets = await api("/api/datasets");
              renderDatasets();
            } catch (err) {
              toast(err.message, true);
              delBtn.disabled = false; delBtn.textContent = "删除";
            }
          },
        },
      ]),
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

async function renderDatasetDetail(id) {
  const data = await api(`/api/datasets/${id}`);
  const info = data.info;
  const rows = data.preview || [];

  clear(canvas);
  removeCompareBar();

  canvas.appendChild(
    el("button", { class: "back-link", onClick: () => navigate("/datasets") }, "← 返回数据列表")
  );

  // 隐藏的文件 input 用于「更新数据」
  const fileInput = el("input", { type: "file", accept: ".csv,.json,.jsonl,.xlsx", style: "display:none" });
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    const updateBtn = actions[0];
    if (updateBtn) { updateBtn.disabled = true; updateBtn.textContent = "上传中…"; }
    const fd = new FormData();
    fd.append("file", file);
    try {
      await api(`/api/datasets/${id}`, { method: "PUT", body: fd });
      toast("数据已更新。");
      await renderDatasetDetail(id);
    } catch (err) {
      toast(err.message, true);
      if (updateBtn) { updateBtn.disabled = false; updateBtn.textContent = "更新数据"; }
    }
    fileInput.value = "";
  });
  canvas.appendChild(fileInput);

  const actions = [
    el("button", { class: "btn", onClick: () => fileInput.click() }, "更新数据"),
    el("a", { class: "btn", href: `/api/datasets/${id}/download` }, "下载"),
    el("button", {
      class: "btn danger",
      onClick: async (e) => {
        if (!confirm(`删除数据集「${info.name}」？`)) return;
        const delBtn = e.currentTarget;
        delBtn.disabled = true; delBtn.textContent = "删除中…";
        try {
          await api(`/api/datasets/${id}`, { method: "DELETE" });
          toast("已删除。");
          navigate("/datasets");
        } catch (err) {
          toast(err.message, true);
          delBtn.disabled = false; delBtn.textContent = "删除";
        }
      },
    }, "删除"),
  ];

  canvas.appendChild(
    el("div", { class: "view-head" }, [
      el("div", {}, [
        el("h1", { class: "view-title" }, info.name),
        el(
          "p",
          { class: "view-sub" },
          `${info.format === "dpo_pairs" ? "偏好 DPO" : "问答 SFT"} · ${info.row_count} 条 · 来自 ${info.source_filename} · ${relativeTime(info.created_at)}`
        ),
      ]),
      el("div", { class: "row-actions" }, actions),
    ])
  );

  // 数据预览表格（先建好，以便诊断行号 chip 联动）
  let previewTable = null;
  if (rows.length) {
    previewTable = buildPreviewTable(rows);
  }

  // 诊断挂载点：先占位、后台拉取，避免阻塞数据预览渲染
  const diagMount = el("div", { class: "data-diag-mount" });
  canvas.appendChild(diagMount);
  api(`/api/datasets/${id}/diagnostics`)
    .then((res) => {
      const cards = (res && res.cards) || [];
      if (!cards.length) { diagMount.remove(); return; }
      diagMount.replaceWith(renderDiagFlowForDetail(cards, previewTable));
    })
    .catch(() => { diagMount.remove(); });

  // 渲染预览
  if (previewTable) {
    canvas.appendChild(el("div", { class: "section-label" }, `数据预览（前 ${rows.length} 条）`));
    canvas.appendChild(previewTable);
  } else {
    canvas.appendChild(el("div", { class: "empty" }, [el("p", { class: "empty-title" }, "这份数据没有可预览的内容")]));
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
  { id: "lively", title: "发散", note: "回答更开放，适合角色和创作", params: { temperature: 1.0, top_p: 0.95, repetition_penalty: 1.2, no_repeat_ngram_size: 3 } },
];

const LAB_PARAM_SPECS = [
  ["temperature", "随机性 temperature", 0, 2, 0.1, "越高越发散，越低越保守。过高会不稳定。"],
  ["top_p", "候选范围 top_p", 0, 1, 0.05, "越小越保守，只在更高概率的词里选择。"],
  ["repetition_penalty", "防复读力度", 1, 1.5, 0.05, "降低重复用词概率。过高会让表达异常。"],
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
    advanced.appendChild(el("summary", {}, "更多设置"));
    clear(paramGrid);
    const values = currentStyleParams();
    for (const [key, label, min, max, step, hint] of LAB_PARAM_SPECS) {
      paramGrid.appendChild(
        buildNumericStepper({
          label,
          min,
          max,
          step,
          value: values[key],
          hint,
          changedText: null,
          onChange: (v) => {
            state.labAdvanced[key] = v;
            paintPresets();
          },
        })
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
  const ids = completed.map((e) => encodeURIComponent(e.id)).join(",");
  try {
    const data = await api(`/api/lab/history/batch?experiment_ids=${ids}&limit=100`);
    return (data.results || []).sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  } catch {
    return [];
  }
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
      actions.length ? el("div", { class: "row-actions lab-head-actions" }, actions) : null,
    ])
  );
}

function renderLabHistoryHome(completed, history) {
  clear(canvas);
  renderLabHeader({
    subtitle: "回看训练前后的回答差异。",
    actions: [el("button", { class: "btn primary", onClick: () => renderLabNew(completed, completed[0].id, history) }, "＋ 新建测评")],
  });

  const experimentOptions = completed;
  if (
    state.labExperimentFilter !== "all" &&
    !experimentOptions.some((exp) => exp.id === state.labExperimentFilter)
  ) {
    state.labExperimentFilter = "all";
  }

  const labFiltersDiv = el("div", { class: "filters" });
  function buildLabFilterBtns() {
    labFiltersDiv.replaceChildren();
    for (const [key, label] of [["all", "全部"], ["compare", "单问对比"], ["batch", "批量测评"], ["chat", "自由对话"]]) {
      labFiltersDiv.appendChild(el("button", {
        class: `filter ${state.labFilter === key ? "active" : ""}`,
        onClick: () => {
          state.labFilter = key;
          buildLabFilterBtns();
          renderLabHistoryRows(list, completed, history);
        },
      }, label));
    }
  }
  buildLabFilterBtns();

  const toolbar = el("div", { class: "view-toolbar" }, [
    labFiltersDiv,
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
    const pollLabHistory = async () => {
      try {
        const freshHistory = await loadAllLabHistory(completed);
        if (parseHash().view === "lab") {
          renderLabHistoryRows(list, completed, freshHistory);
          if (freshHistory.some((item) => labResultStatus(item) === "running")) {
            state.pollTimer = setTimeout(pollLabHistory, 2500);
          }
        }
      } catch { /* ignore */ }
    };
    state.pollTimer = setTimeout(pollLabHistory, 2500);
  }
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
  const summary = labHistorySummary(item);
  return el("div", { class: "lab-history-row clickable", onClick: () => renderLabDetail(item, completed, history) }, [
    el("div", { class: "lab-history-main" }, [
      el("span", { class: "lab-history-title" }, item.kind === "batch" ? item.prompt || "批量测评" : item.kind === "chat" ? item.prompt || "自由对话" : item.prompt),
      el("span", { class: "lab-history-meta" }, [
        kindLabel,
        ` · ${labExperimentName(item.experiment_id)}`,
        item.kind === "chat" ? ` · ${(item.data?.messages || []).filter((m) => m.role === "user").length} 轮` : ` · ${questionCount} 条`,
        status === "running" ? " · 测评中" : "",
        status === "failed" ? " · 失败" : "",
        ` · ${formatDateTime(item.created_at)}`,
      ]),
      summary ? el("span", { class: "lab-history-summary" }, summary) : null,
    ]),
    actionMenu([
      {
        label: "删除记录",
        danger: true,
        onClick: async (e) => {
          if (!confirm("删除这条测评记录？")) return;
          const delBtn = e.currentTarget;
          delBtn.disabled = true; delBtn.textContent = "删除中…";
          try {
            await api(`/api/lab/history/${item.id}`, { method: "DELETE" });
            await renderLab();
          } catch (err) {
            toast(err.message, true);
            delBtn.disabled = false; delBtn.textContent = "删除";
          }
        },
      },
    ]),
  ]);
}

function compactText(text, limit = 72) {
  if (!text) return "";
  const clean = String(text).replace(/\s+/g, " ").trim();
  return clean.length > limit ? `${clean.slice(0, limit)}...` : clean;
}

function labHistorySummary(item) {
  const status = labResultStatus(item);
  if (status === "running") return "正在生成训练前后回答";
  if (status === "failed") return item.data?.error || "测评失败";
  if (item.kind === "chat") {
    const rounds = (item.data?.messages || []).filter((m) => m.role === "user").length;
    return rounds ? `${rounds} 轮对话` : "没有保存对话内容";
  }
  const rows = labResultItems(item);
  const first = rows[0] || {};
  const after = compactText(first.finetuned_answer, 64);
  if (item.kind === "batch") {
    return after ? `训练后示例：${after}` : `${rows.length || 0} 条问题`;
  }
  return after ? `训练后：${after}` : "";
}

async function renderLabNew(completed, initialExperimentId, history = []) {
  clear(canvas);
  let activeExperimentId = initialExperimentId;
  let mode = "compare";
  let batchSource = "sample";

  if (history.length) {
    canvas.appendChild(
      el("button", { class: "back-link", onClick: () => renderLabHistoryHome(completed, history) }, "← 返回测评历史")
    );
  }

  const select = el(
    "select",
    { class: "lab-select" },
    completed.map((e) => el("option", { value: e.id, selected: activeExperimentId === e.id }, e.name))
  );
  const targetLine = el("div", { class: "lab-target" });
  const form = el("div", { class: "lab-new-form" });

  renderLabHeader({
    subtitle: history.length ? "选择训练结果，用一组问题验证训练前后的变化。" : "第一次测评：选择一个已完成的训练结果开始验证。",
    actions: [],
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
        el("span", { class: "choice-note" }, "连续追问，补充判断训练效果"),
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

function renderLabDetail(item, completed, history, { animate = true } = {}) {
  clear(canvas, { animate });
  if (!item) {
    renderLabHistoryHome(completed, history);
    return;
  }

  // 对话类型使用专门的回看视图
  if (item.kind === "chat") {
    renderLabChatDetail(item, completed, history);
    return;
  }

  canvas.appendChild(
    el("button", { class: "back-link", onClick: () => renderLabHistoryHome(completed, history) }, "← 返回测评历史")
  );

  const rows = labResultItems(item);
  const typeLabel = item.kind === "batch" ? "批量测评" : "单问对比";
  const status = labResultStatus(item);
  renderLabHeader({
    subtitle: `${typeLabel} · ${labExperimentName(item.experiment_id)} · ${formatDateTime(item.created_at)}`,
    actions: [
      el("button", { class: "btn primary", onClick: () => renderLabNew(completed, item.experiment_id, history) }, "＋ 新建测评"),
      el("button", {
        class: "btn danger",
        onClick: async (e) => {
          if (!confirm("删除这条测评记录？")) return;
          const delBtn = e.currentTarget;
          delBtn.disabled = true; delBtn.textContent = "删除中…";
          try {
            await api(`/api/lab/history/${item.id}`, { method: "DELETE" });
            toast("已删除。");
            renderLabHistoryHome(completed, await loadAllLabHistory(completed));
          } catch (err) {
            toast(err.message, true);
            delBtn.disabled = false; delBtn.textContent = "删除";
          }
        },
      }, "删除"),
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
        renderLabDetail(fresh, completed, await loadAllLabHistory(completed), { animate: false });
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

  canvas.appendChild(
    el("button", { class: "back-link", onClick: () => renderLabHistoryHome(completed, history) }, "← 返回测评历史")
  );

  renderLabHeader({
    subtitle: `自由对话 · ${labExperimentName(item.experiment_id)} · ${formatDateTime(item.created_at)}`,
    actions: [
      el("button", {
        class: "btn danger",
        onClick: async (e) => {
          if (!confirm("删除这条测评记录？")) return;
          const delBtn = e.currentTarget;
          delBtn.disabled = true; delBtn.textContent = "删除中…";
          try {
            await api(`/api/lab/history/${item.id}`, { method: "DELETE" });
            toast("已删除。");
            renderLabHistoryHome(completed, await loadAllLabHistory(completed));
          } catch (err) {
            toast(err.message, true);
            delBtn.disabled = false; delBtn.textContent = "删除";
          }
        },
      }, "删除"),
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
  canvas.classList.add("chat-main");
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
    const btn = document.getElementById("chatEnd");
    if (btn) { btn.disabled = true; btn.textContent = "保存中…"; }
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
    const btn = document.getElementById("chatRelease");
    if (btn) { btn.disabled = true; btn.textContent = "释放中…"; }
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
preloadMeta();
render();
