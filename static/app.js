const $ = (id) => document.getElementById(id);
const state = {
  catalog: {roles: {}, tools: [], review_states: {}},
  panels: [], panel: null, captures: [], capture: null,
  calibrations: [], compareCalibrations: [], region: null,
  edit: null, cornerDraft: [], startProbe: null,
  draftResult: null, activeTab: "capture",
  records: [], record: null, checkedRecords: new Set(),
  requestSerial: 0, recordSerial: 0, currentSearch: 0,
};

function node(tag, text, attrs = {}) {
  const el = document.createElement(tag);
  if (text !== undefined && text !== null) el.textContent = text;
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}

function notice(text, error = false) {
  $("notice").textContent = text;
  $("notice").hidden = !text;
  $("notice").classList.toggle("error", error);
}

async function json(path, options = {}) {
  const init = {...options};
  if (init.body && !(init.body instanceof FormData)) {
    init.body = JSON.stringify(init.body);
    init.headers = {"Content-Type": "application/json", ...init.headers};
  }
  const response = await fetch(path, init);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`请求失败（${response.status}）`);
  }
  if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
  return data;
}

const post = (path, body = {}) => json(path, {method: "POST", body});
const escapePath = encodeURIComponent;
function on(id, event, handle) {
  $(id).addEventListener(event, (ev) => {
    Promise.resolve().then(() => handle(ev)).catch((error) => notice(error.message, true));
  });
}

function formValues(form) {
  const values = Object.fromEntries(new FormData(form));
  for (const check of form.querySelectorAll("input[type=checkbox]")) values[check.name] = check.checked;
  return values;
}

function setOptions(select, values, fallback, placeholder) {
  const previous = select.value;
  select.replaceChildren();
  if (placeholder) select.append(node("option", placeholder, {value: ""}));
  for (const item of values) select.append(node("option", item[1], {value: item[0]}));
  const allowed = new Set([...select.options].map((option) => option.value));
  if (previous && allowed.has(previous)) select.value = previous;
  else if (fallback !== undefined && allowed.has(fallback)) select.value = fallback;
}

function listEntry(title, subtitle, active, callback) {
  const button = node("button", null, {type: "button", class: "lib-item", "aria-current": String(active)});
  button.append(node("b", title), node("small", subtitle));
  button.addEventListener("click", () => callback().catch((error) => notice(error.message, true)));
  return button;
}

function displayPanels() {
  const root = $("panels");
  root.replaceChildren();
  for (const panel of state.panels) {
    const info = [panel.station || "未填写工位", `${panel.capture_count} 张采集`, panel.synthetic ? "合成" : ""]
      .filter(Boolean).join(" · ");
    root.append(listEntry(panel.name, info, state.panel?.id === panel.id, () => selectPanel(panel)));
  }
  if (!root.children.length) root.append(node("p", "没有符合筛选的面板。", {class: "library-note"}));
}

function displayCaptureList() {
  const root = $("captures");
  root.replaceChildren();
  if (!state.panel) return;
  const selected = $("capture-filter").value;
  const filtered = state.captures.filter((capture) => !selected || selected === capture.role);
  for (const capture of filtered) {
    const revision = capture.calibration ? `标定 v${capture.calibration.revision}` : "整图坐标";
    const info = `${state.catalog.roles[capture.role]} · ${capture.width} × ${capture.height} · ${revision}`;
    root.append(listEntry(capture.title, info, capture.id === state.capture?.id, () => selectCapture(capture)));
  }
  if (!root.children.length) root.append(node("p", "当前筛选下没有采集。", {class: "library-note"}));
}

async function refreshPanels() {
  const search = ++state.currentSearch;
  const q = escapePath($("panel-search").value);
  const payload = await json(`/api/panels?q=${q}`);
  if (search !== state.currentSearch) return;
  state.panels = payload.panels;
  const current = state.panels.find((p) => p.id === state.panel?.id);
  if (current) state.panel = current;
  displayPanels();
}

function displayPanel() {
  $("empty").hidden = Boolean(state.panel);
  $("workbench").hidden = !state.panel;
  $("capture-section").hidden = !state.panel;
  if (!state.panel) return;
  const panel = state.panel;
  $("station").textContent = panel.station || "未填写工位";
  $("panel-title").textContent = panel.name;
  $("synthetic").hidden = !panel.synthetic;
  const dimensions = panel.width_mm ? ` · ${panel.width_mm} × ${panel.height_mm} mm` : "";
  $("panel-summary").textContent = `${panel.native_width} × ${panel.native_height} px${dimensions} · ${panel.capture_count} 张采集 · ${panel.record_count} 条记录`;
  $("bundle-form").action = `/api/panels/${panel.id}/handoff`;
  $("fill-demo-log").hidden = !panel.synthetic;
}

async function selectPanel(panel, preferred = null) {
  state.panel = panel;
  state.capture = null;
  state.captures = [];
  state.record = null;
  state.checkedRecords.clear();
  state.activeTab = "capture";
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === "capture");
  });
  document.querySelectorAll(".tab-panel").forEach((part) => {
    part.hidden = part.id !== "tab-capture";
  });
  localStorage.setItem("lumagrid.panel", panel.id);
  $("capture-filter").value = "";
  $("record-detail").replaceChildren(node("p", "选择一条记录，查看当时的图像和标定版本。", {class: "muted"}));
  for (const id of ["report", "timing-report", "flatfield-report"]) $(id).hidden = true;
  displayPanel();
  displayPanels();
  displayCaptureList();
  displayCapture();
  const payload = await json(`/api/panels/${panel.id}/captures`);
  if (state.panel?.id !== panel.id) return;
  state.captures = payload.captures;
  displayCaptureList();
  const starting = state.captures.find((capture) => capture.id === preferred) ||
    state.captures.find((capture) => capture.filename === "synthetic-uniform.png") || state.captures[0];
  if (starting) await selectCapture(starting);
  fillCorrectionFields();
}

async function refreshCaptures() {
  if (!state.panel) return;
  const panelId = state.panel.id;
  const payload = await json(`/api/panels/${panelId}/captures`);
  if (state.panel?.id !== panelId) return;
  state.captures = payload.captures;
  displayCaptureList();
  fillCorrectionFields();
  fillComparison();
  if (state.activeTab === "intake") packetMembers();
}

async function selectCapture(capture) {
  state.capture = capture;
  state.calibrations = [];
  state.region = null;
  state.edit = null;
  state.cornerDraft = [];
  state.startProbe = null;
  state.draftResult = null;
  $("report").hidden = true;
  $("save-report").disabled = true;
  $("revision").replaceChildren();
  displayCapture();
  displayCaptureList();
  fillComparison();
  localStorage.setItem("lumagrid.capture", capture.id);
  const payload = await json(`/api/captures/${capture.id}/calibrations`);
  if (state.capture?.id !== capture.id) return;
  setCalibrationList(payload.calibrations);
}

function displayCapture() {
  const capture = state.capture;
  $("no-capture").hidden = Boolean(capture);
  $("image-wrap").hidden = !capture;
  $("stage-control").hidden = !capture;
  $("download-original").hidden = !capture;
  $("tool-panel").hidden = !capture;
  $("capture-title").textContent = capture ? capture.title : "相机视图";
  $("capture-meta").textContent = capture ?
    [`${capture.width} × ${capture.height} px`, state.catalog.roles[capture.role],
     capture.exposure.camera, capture.exposure.shutter_ms ? `${capture.exposure.shutter_ms} ms` : null]
      .filter(Boolean).join(" · ") : "";
  if (capture) {
    const base = `/api/captures/${capture.id}/image`;
    $("camera-image").src = `${base}/preview`;
    $("download-original").href = `${base}/original`;
    $("download-original").download = capture.filename;
  } else {
    $("camera-image").removeAttribute("src");
  }
  drawOverlay();
}

function revisionChoices(entries) {
  return entries.map((entry) => [entry.id, `v${entry.revision} · 覆盖 ${entry.info.coverage_pct}%`])
    .concat([["identity", "整张相机图"]]);
}

function setCalibrationList(entries, desired) {
  state.calibrations = entries;
  setOptions($("revision"), revisionChoices(entries), entries[0]?.id || "identity");
  if (desired && [...$("revision").options].some((option) => option.value === desired)) {
    $("revision").value = desired;
  }
  drawOverlay();
}

const UNIT = [[0, 0], [1, 0], [1, 1], [0, 1]];
function solve(matrix) {
  for (let i = 0; i < matrix.length; i++) {
    let pivot = i;
    for (let j = i + 1; j < matrix.length; j++) if (Math.abs(matrix[j][i]) > Math.abs(matrix[pivot][i])) pivot = j;
    [matrix[i], matrix[pivot]] = [matrix[pivot], matrix[i]];
    const base = matrix[i][i];
    if (Math.abs(base) < 1e-10) return null;
    matrix[i] = matrix[i].map((v) => v / base);
    for (let j = 0; j < matrix.length; j++) {
      if (j === i) continue;
      const factor = matrix[j][i];
      matrix[j] = matrix[j].map((v, k) => v - factor * matrix[i][k]);
    }
  }
  return matrix.map((row) => row.at(-1));
}

function homography(from, to) {
  return solve(from.flatMap(([x, y], i) => {
    const [u, v] = to[i];
    return [[x, y, 1, 0, 0, 0, -u * x, -u * y, u], [0, 0, 0, x, y, 1, -v * x, -v * y, v]];
  }));
}

function remap(matrix, [x, y]) {
  if (!matrix) return [x, y];
  const scale = matrix[6] * x + matrix[7] * y + 1;
  return [(matrix[0] * x + matrix[1] * y + matrix[2]) / scale,
          (matrix[3] * x + matrix[4] * y + matrix[5]) / scale];
}

function cornerPoints() {
  return state.calibrations.find((entry) => entry.id === $("revision").value)?.corners || UNIT;
}

function svgEl(tag, attributes) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
  return el;
}

function drawOverlay() {
  const svg = $("image-overlay");
  svg.replaceChildren();
  $("calibration-edit").hidden = state.edit !== "corners";
  $("save-corners").disabled = state.cornerDraft.length !== 4;
  $("image-wrap").classList.toggle("editing", Boolean(state.edit));
  $("edit-corners").setAttribute("aria-pressed", String(state.edit === "corners"));
  $("select-region").setAttribute("aria-pressed", String(state.edit === "region"));
  if (!state.capture) {
    $("region-readout").hidden = true;
    return;
  }
  const points = state.edit === "corners" ? state.cornerDraft : cornerPoints();
  const csv = (values) => values.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ");
  if (points.length > 1) svg.append(svgEl(points.length === 4 ? "polygon" : "polyline", {points: csv(points)}));
  if (state.edit === "corners") {
    for (const [x, y] of points) svg.append(svgEl("circle", {cx: x * 1000, cy: y * 1000, r: 7}));
  }
  if (state.region) {
    const [a, b, c, d] = state.region;
    const transform = toolSpec()?.coordinate_space === "camera" ? null : homography(UNIT, cornerPoints());
    const selection = [[a, b], [c, b], [c, d], [a, d]].map((point) => remap(transform, point));
    const rect = svgEl("polygon", {points: csv(selection), "stroke-dasharray": "6 4"});
    svg.append(rect);
  }
  const message = state.edit === "region" ? "在画面拖动以选取范围，或连续点选两端。" :
    state.region ? `面板选区 ${state.region.map((x) => x.toFixed(3)).join(" / ")} · ${toolSpec()?.coordinate_space === "camera" ? "相机" : "面板"}归一化坐标` : "分析范围：整张面板";
  $("region-readout").hidden = false;
  $("region-readout").textContent = message;
}

function pointerPoint(event) {
  const bounds = $("camera-image").getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
    Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
  ];
}

function setRegion(first, last) {
  const inverse = toolSpec()?.coordinate_space === "camera" ? null : homography(cornerPoints(), UNIT);
  const [a, b] = remap(inverse, first).map((x) => Math.max(0, Math.min(1, x)));
  const [c, d] = remap(inverse, last).map((x) => Math.max(0, Math.min(1, x)));
  const output = [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)];
  if (output[2] - output[0] < .01 || output[3] - output[1] < .01) {
    notice("选区至少要覆盖面板宽高的 1%。", true);
    return;
  }
  state.region = output;
  state.edit = null;
  state.startProbe = null;
  invalidateDraft();
  drawOverlay();
  notice("");
}

let pointerStart;
$("image-wrap").addEventListener("pointerdown", (ev) => {
  if (!state.edit || ev.button !== 0) return;
  ev.preventDefault();
  pointerStart = pointerPoint(ev);
  $("image-wrap").setPointerCapture(ev.pointerId);
});
$("image-wrap").addEventListener("pointerup", (ev) => {
  if (!state.edit || !pointerStart) return;
  const previous = pointerStart;
  pointerStart = null;
  const p = pointerPoint(ev);
  if (state.edit === "corners") {
    if (state.cornerDraft.length === 4) state.cornerDraft = [];
    state.cornerDraft.push(p);
    drawOverlay();
  } else if (Math.hypot(p[0] - previous[0], p[1] - previous[1]) > .012) {
    setRegion(previous, p);
  } else if (state.startProbe) {
    setRegion(state.startProbe, p);
  } else {
    state.startProbe = p;
  }
});
$("image-wrap").addEventListener("pointercancel", () => { pointerStart = null; });

function toolSpec() {
  return state.catalog.tools.find((spec) => spec.key === $("tool").value);
}

function fillComparison() {
  const compare = state.captures.filter((cap) => cap.id !== state.capture?.id);
  setOptions($("comparison"), compare.map((item) => [item.id, item.title]), undefined, "选择现有采集");
}

async function updateCompareCalibrations() {
  const id = $("comparison").value;
  const enabled = Boolean(id) && Boolean(toolSpec()?.comparison);
  $("compare-revision-field").hidden = !enabled;
  if (!enabled) return;
  const {calibrations} = await json(`/api/captures/${id}/calibrations`);
  if ($("comparison").value !== id || !toolSpec()?.comparison) return;
  state.compareCalibrations = calibrations;
  $("compare-revision").replaceChildren();
  setOptions($("compare-revision"), revisionChoices(calibrations), calibrations[0]?.id || "identity");
}

function buildField(config) {
  const {label, type, key, default: initial} = config;
  const wrap = node("label", label);
  let field;
  if (type === "select") {
    field = node("select");
    setOptions(field, Object.entries(config.values), initial);
  } else if (type === "json") {
    field = node("textarea", null, {rows: "5", spellcheck: "false"});
    field.value = JSON.stringify(initial ?? null, null, 2);
  } else {
    field = node("input", null, {type});
    if (type === "checkbox") {
      wrap.className = "check-line";
      field.checked = Boolean(initial);
    } else {
      for (const property of ["min", "max", "step"]) if (config[property] !== undefined) {
        field.setAttribute(property, String(config[property]));
      }
      field.value = initial ?? "";
    }
  }
  field.name = key;
  field.dataset.valueType = type;
  if (type === "checkbox") wrap.prepend(field);
  else wrap.append(field);
  return wrap;
}

async function displayTool() {
  const tool = toolSpec();
  if (!tool) return;
  $("tool-description").textContent = tool.description;
  $("compare-field").hidden = !tool.comparison;
  $("options").replaceChildren(...tool.options.map(buildField));
  state.region = null;
  drawOverlay();
  fillComparison();
  invalidateDraft();
  await updateCompareCalibrations();
}

function invalidateDraft() {
  state.draftResult = null;
  state.requestSerial++;
  $("save-report").disabled = true;
}

function currentRequest() {
  if (!state.capture) throw new Error("请先选择采集图。");
  const tool = toolSpec();
  const options = {};
  for (const field of $("options").querySelectorAll("[data-value-type]")) {
    const type = field.dataset.valueType;
    if (type === "checkbox") options[field.name] = field.checked;
    else if (type === "json") {
      try {
        options[field.name] = JSON.parse(field.value);
      } catch {
        throw new Error("检查点数组不是有效 JSON。");
      }
    } else if (type === "number") {
      if (!field.checkValidity()) {
        field.reportValidity();
        throw new Error("检查工具参数超出可用范围。");
      }
      options[field.name] = field.value === "" ? null : field.valueAsNumber;
    } else {
      options[field.name] = field.value;
    }
  }
  const output = {
    tool: tool.key,
    region: state.region,
    calibration_id: $("revision").value || "identity",
    options,
  };
  if (tool.comparison) {
    output.comparison = $("comparison").value;
    output.comparison_calibration_id = $("compare-revision").value || "identity";
    if (!output.comparison && tool.comparison === "required") throw new Error("请先选择对照采集。");
  }
  return output;
}

async function runReport(save) {
  const serial = ++state.requestSerial;
  const capture = state.capture;
  const body = save && state.draftResult ? state.draftResult : currentRequest();
  const focused = save ? $("save-report") : $("preview");
  focused.disabled = true;
  notice(save ? "正在保存本次检查…" : "正在处理采集图…");
  try {
    const reply = await post(`/api/captures/${capture.id}/analysis`, {...body, save});
    if (serial !== state.requestSerial || capture.id !== state.capture?.id) return;
    showReport($("report"), reply.result || reply);
    state.draftResult = body;
    $("save-report").disabled = save;
    notice(save ? "检查已保存，可以在复核与交接中打开。" : "");
    if (save) {
      await refreshPanels();
      displayPanel();
    }
  } finally {
    if (save) $("save-report").disabled = !state.draftResult;
    else $("preview").disabled = false;
  }
}

const labels = {
  region: "归一化选区", working_size: "采样尺寸", coverage_pct: "有效覆盖 %", positive_samples: "有效采样",
  source_title: "主采集", synthetic: "合成材料", row: "行", column: "列", samples: "样本", mean: "均值",
  p05: "第 5 百分位", p95: "第 95 百分位", median: "中位数", stddev: "标准差", cells_used: "纳入网格",
  cells_total: "全部网格", min_p50_over_max_p50: "最暗格与最亮格之比", relative_reference: "相对参考码值",
  dimmest_cell: "最暗网格", brightest_cell: "最亮网格", scale: "采用尺度", heatmap_range: "热图标尺",
  contrast_ratio: "亮暗比", michelson: "米氏对比度", white_cells: "亮格数", black_cells: "暗格数",
  white_level: "亮格中位数", black_level: "暗格中位数", mtf50_cycles_per_sample: "MTF50 周期／采样",
  mtf10_cycles_per_sample: "MTF10 周期／采样", dark_title: "暗场", flat_title: "平场",
  source_covered_pct: "校正覆盖 %", clipped_pct: "裁切 %", matched_pairs: "配对数", clock_offset_ms: "时钟偏移毫秒",
  unmatched_frames: "未匹配采集", unmatched_reference: "未匹配事件", rms_residual_ms: "对时均方根毫秒",
  dry_patches: "样本格数", source_error_x_px: "横向误差 px", source_error_y_px: "纵向误差 px",
  panel_mm: "面板毫米长度", panel_px: "面板物理像素长度", source_px: "原相机像素长度",
  dark_capture_id: "暗场标识", flat_capture_id: "平场标识", fit_rmse: "拟合均方根误差",
};

function valueText(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) return value.map(valueText).join(" / ");
  if (typeof value === "object") return Object.entries(value).map(([key, inner]) => `${labels[key] || key} · ${valueText(inner)}`).join("，");
  return String(value);
}

function showReport(destination, report, savedRecord) {
  destination.replaceChildren();
  const header = node("header");
  header.append(node("h2", report.title), node("p", report.summary));
  destination.append(header);
  if (report.metrics) {
    const matrix = node("dl", null, {class: "metric-grid"});
    for (const [key, value] of Object.entries(report.metrics)) {
      const cell = node("div");
      cell.append(node("dt", labels[key] || key.replaceAll("_", " ")), node("dd", valueText(value)));
      matrix.append(cell);
    }
    destination.append(matrix);
  }
  if (report.warnings?.length) {
    const warnings = node("ul", null, {class: "warning-list"});
    warnings.append(...report.warnings.map((item) => node("li", item)));
    destination.append(warnings);
  }
  const images = node("div", null, {class: "report-gallery"});
  const files = node("div", null, {class: "inline-actions"});
  for (const attachment of report.files || []) {
    const mime = attachment.mime;
    const url = savedRecord ? `/api/records/${savedRecord}/files/${escapePath(attachment.name)}` : undefined;
    const preview = attachment.preview || (mime.startsWith("image/") ? url : "");
    if (preview) {
      const figure = node("figure");
      const pixels = node("img", null, {src: preview, alt: attachment.name, loading: "lazy"});
      const caption = node("figcaption", attachment.name);
      if (url) {
        caption.append(document.createTextNode(" · "), node("a", "下载", {href: `${url}?download=1`, download: attachment.name}));
      }
      figure.append(pixels, caption);
      images.append(figure);
    } else if (url) {
      files.append(node("a", attachment.name, {href: `${url}?download=1`, download: attachment.name, class: "button compact quiet"}));
    }
  }
  if (images.children.length) destination.append(images);
  if (files.children.length) destination.append(files);
  if (report.rows?.length) {
    const details = node("details", null, {class: "report-details"});
    details.append(node("summary", `明细数据（${report.rows.length} 行）`));
    const wrap = node("div", null, {class: "data-table"});
    const table = node("table");
    const cols = [...new Set(report.rows.flatMap(Object.keys))];
    const head = node("thead");
    const tr = node("tr");
    tr.append(...cols.map((key) => node("th", labels[key] || key.replaceAll("_", " "), {scope: "col"})));
    head.append(tr);
    const entries = node("tbody");
    for (const record of report.rows) {
      const line = node("tr");
      line.append(...cols.map((key) => node("td", valueText(record[key]))));
      entries.append(line);
    }
    table.append(head, entries);
    wrap.append(table);
    details.append(wrap);
    destination.append(details);
  }
  destination.hidden = false;
}

function fillCorrectionFields() {
  const form = $("flatfield-form");
  const entries = (list) => list.map((capture) => [capture.id, capture.title]);
  const primary = state.captures.filter((item) => !["dark", "flat"].includes(item.role));
  setOptions(form.elements.source, entries(primary), state.capture?.id, "选择主采集");
  setOptions(form.elements.dark, entries(state.captures.filter((item) => item.role === "dark")), undefined, "选择暗场");
  setOptions(form.elements.flat, entries(state.captures.filter((item) => item.role === "flat")), undefined, "选择平场");
}

function syncBundleSelection() {
  const form = $("bundle-form");
  form.querySelectorAll("input[name=record]").forEach((input) => input.remove());
  for (const id of state.checkedRecords) form.append(node("input", null, {type: "hidden", name: "record", value: id}));
  $("bundle-download").disabled = !state.checkedRecords.size || state.checkedRecords.size > 18;
}

function displayRecords() {
  const root = $("records");
  root.replaceChildren();
  for (const record of state.records) {
    const row = node("div", null, {class: `record-option${record.id === state.record?.id ? " current" : ""}`});
    const checkbox = node("input", null, {type: "checkbox", "aria-label": `选择交接 ${record.title}`});
    checkbox.disabled = record.state !== "verified";
    checkbox.checked = state.checkedRecords.has(record.id) && !checkbox.disabled;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.checkedRecords.add(record.id);
      else state.checkedRecords.delete(record.id);
      syncBundleSelection();
    });
    const open = node("button", record.title, {type: "button", class: "button quiet", "aria-label": `查看 ${record.title}`});
    open.append(node("small", state.catalog.review_states[record.state] || record.state));
    open.addEventListener("click", () => selectRecord(record.id).catch((error) => notice(error.message, true)));
    row.append(checkbox, open);
    root.append(row);
  }
  if (!root.children.length) root.append(node("p", "没有符合筛选的检查记录。", {class: "muted"}));
  syncBundleSelection();
}

async function refreshRecords() {
  const panelId = state.panel?.id;
  if (!panelId) return;
  const serial = ++state.recordSerial;
  const filter = escapePath($("record-filter").value);
  const {records} = await json(`/api/panels/${panelId}/records?state=${filter}`);
  if (serial !== state.recordSerial || state.panel?.id !== panelId) return;
  state.records = records;
  for (const id of state.checkedRecords) {
    if (records.some((record) => record.id === id && record.state !== "verified")) state.checkedRecords.delete(id);
  }
  displayRecords();
}

function blockLabel(text, child) {
  const label = node("label", text);
  label.append(child);
  return label;
}

async function selectRecord(id) {
  const panel = state.panel?.id;
  const payload = await json(`/api/records/${id}`);
  if (state.panel?.id !== panel) return;
  const {record, decisions} = payload;
  state.record = record;
  displayRecords();
  const detail = $("record-detail");
  detail.replaceChildren();
  const summary = node("header");
  const current = state.catalog.review_states[record.state] || record.state;
  const rev = record.request?.source?.calibration_revision;
  const compareRev = record.request?.comparison?.calibration_revision;
  const source = record.request?.source?.capture_title;
  const matters = [current, source, rev ? `主图标定 v${rev}` : "主图整图坐标", compareRev ? `对照标定 v${compareRev}` : ""];
  summary.append(node("h2", record.title), node("p", matters.filter(Boolean).join(" · "), {class: "muted"}));
  detail.append(summary);
  if (decisions.length) {
    const log = node("ul", null, {class: "decision-log"});
    for (const row of decisions) {
      log.append(node("li", [row.operator, state.catalog.review_states[row.new_state], row.note].filter(Boolean).join(" · ")));
    }
    detail.append(log);
  }
  const form = node("form", null, {class: "decision-form"});
  const reviewer = node("input", null, {name: "operator", maxlength: 80, required: ""});
  const outcome = node("select", null, {name: "state"});
  setOptions(outcome, Object.entries(state.catalog.review_states).filter(([key]) => key !== record.state), "verified");
  const notes = node("input", null, {name: "note", maxlength: 600});
  form.append(
    blockLabel("复核人", reviewer), blockLabel("复核决定", outcome), blockLabel("复核说明", notes),
    node("button", "保存复核", {type: "submit", class: "button primary"}),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const data = formValues(form);
      data.expected = record.state;
      await post(`/api/records/${id}/decisions`, data);
      await refreshRecords();
      await selectRecord(id);
      notice("复核状态已更新。");
    } catch (error) {
      notice(error.message, true);
    }
  });
  detail.append(form);
  const result = node("article", null, {class: "report"});
  detail.append(result);
  showReport(result, {...record.report, files: record.files}, record.id);
}

async function installDemo() {
  $("load-demo").disabled = true;
  $("empty-demo").disabled = true;
  notice("正在载入合成演示采集…");
  try {
    const panel = await post("/api/demo");
    $("panel-search").value = "";
    await refreshPanels();
    await selectPanel(panel);
    notice("");
  } finally {
    $("load-demo").disabled = false;
    $("empty-demo").disabled = false;
  }
}

on("empty-demo", "click", installDemo);
on("load-demo", "click", installDemo);
on("panel-search", "input", refreshPanels);
on("capture-filter", "input", displayCaptureList);
on("record-filter", "input", refreshRecords);
on("tool", "change", displayTool);
on("options", "input", invalidateDraft);
on("comparison", "change", async () => { invalidateDraft(); await updateCompareCalibrations(); });
on("compare-revision", "change", invalidateDraft);
on("revision", "change", () => { invalidateDraft(); drawOverlay(); });
on("preview", "click", () => runReport(false));
on("save-report", "click", () => runReport(true));
on("new-panel", "click", () => $("panel-dialog").showModal());
on("new-capture", "click", () => $("capture-dialog").showModal());
document.querySelectorAll("[data-close]").forEach((button) => {
  button.addEventListener("click", () => $(button.dataset.close).close());
});
on("panel-form", "submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const panel = await post("/api/panels", formValues(form));
  $("panel-dialog").close();
  form.reset();
  $("panel-search").value = "";
  await refreshPanels();
  await selectPanel(panel);
  notice("面板已建立。");
});
on("capture-form", "submit", async (event) => {
  event.preventDefault();
  if (!state.panel) return;
  const form = event.target;
  const capture = await post(`/api/panels/${state.panel.id}/captures`, new FormData(form));
  $("capture-dialog").close();
  form.reset();
  $("capture-filter").value = "";
  await refreshPanels();
  await refreshCaptures();
  displayPanel();
  await selectCapture(capture);
  notice("采集已导入。");
});
on("edit-corners", "click", () => {
  state.edit = "corners";
  state.cornerDraft = [];
  state.startProbe = null;
  $("calibration-note").value = "";
  drawOverlay();
});
on("cancel-corners", "click", () => {
  state.edit = null;
  drawOverlay();
});
on("save-corners", "click", async () => {
  const current = state.capture;
  const rev = await post(`/api/captures/${current.id}/calibrations`, {
    corners: state.cornerDraft,
    note: $("calibration-note").value,
  });
  if (state.capture?.id !== current.id) return;
  state.edit = null;
  state.region = null;
  await refreshCaptures();
  const payload = await json(`/api/captures/${current.id}/calibrations`);
  setCalibrationList(payload.calibrations, rev.id);
  invalidateDraft();
  notice(`已保存标定 v${rev.revision}。`);
});
on("select-region", "click", () => {
  state.edit = state.edit === "region" ? null : "region";
  state.startProbe = null;
  drawOverlay();
});
on("clear-region", "click", () => {
  state.region = null;
  state.edit = null;
  state.startProbe = null;
  drawOverlay();
  invalidateDraft();
});
on("fill-demo-log", "click", async () => {
  const result = await fetch("/api/demo/reference-log");
  if (!result.ok) throw new Error("读取合成参考日志失败。");
  $("timing-form").elements.reference_csv.value = await result.text();
});
on("timing-form", "submit", async (event) => {
  event.preventDefault();
  const body = formValues(event.target);
  body.save = event.submitter?.name === "save";
  const output = await post(`/api/panels/${state.panel.id}/timing`, body);
  showReport($("timing-report"), output.result || output, output.record?.id);
  if (body.save) {
    await refreshPanels();
    displayPanel();
    notice("时钟配对记录已保存。");
  }
});
on("flatfield-form", "submit", async (event) => {
  event.preventDefault();
  const data = formValues(event.target);
  if (!data.source || !data.dark || !data.flat) throw new Error("请先选择主采集、暗场和平场。");
  data.save = event.submitter?.name === "save";
  const output = await post(`/api/captures/${data.source}/flatfield`, data);
  showReport($("flatfield-report"), output.result || output);
  if (data.save) {
    await refreshPanels();
    await refreshCaptures();
    displayPanel();
    notice("已将校正结果存为派生采集。");
  }
});
on("verify-form", "submit", async (event) => {
  event.preventDefault();
  const result = await post("/api/handoff/inspect", new FormData(event.target));
  $("verify-result").textContent = result.valid ? `已核对 ${result.checked_files} 个文件，哈希一致。` :
    `${result.failed.length} 个文件与清单不符。`;
});
document.querySelectorAll("[data-tab]").forEach((tab) => {
  tab.addEventListener("click", async () => {
    state.activeTab = tab.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach((button) => button.classList.toggle("active", button === tab));
    document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== `tab-${state.activeTab}`; });
    if (state.activeTab === "records") {
      try { await refreshRecords(); }
      catch (error) { notice(error.message, true); }
    } else if (state.activeTab === "backdrop") fillCorrectionFields();
    else if (state.activeTab === "intake") packetMembers();
  });
});


function packetMembers() {
  const root = $("packet-members");
  const exportForm = $("packet-export");
  exportForm.action = `/api/panels/${state.panel.id}/capture-packet`;
  const seen = new Set([...root.querySelectorAll(":checked")].map((option) => option.value));
  root.replaceChildren();
  for (const capture of state.captures) {
    const checkbox = node("input", null, {type: "checkbox", name: "selected", value: capture.id});
    checkbox.checked = seen.has(capture.id);
    const text = node("span", capture.title);
    const ui = node("label", null, {class: "check-line"});
    ui.append(checkbox, text);
    checkbox.addEventListener("change", () => {
      const selected = root.querySelectorAll(":checked").length;
      $("packet-download").disabled = selected === 0 || selected > 20;
    });
    root.append(ui);
  }
  $("packet-download").disabled = root.querySelectorAll(":checked").length === 0;
}

on("packet-import", "input", (event) => {
  $("packet-report").hidden = true;
  $("packet-import").elements.save.disabled = true;
  delete event.target.form?.dataset.previewHash;
});
on("packet-import", "submit", async (event) => {
  event.preventDefault();
  const panelId = state.panel.id;
  const data = new FormData(event.target);
  const save = event.submitter?.name === "save";
  data.set("save", save ? "1" : "0");
  if (save) data.set("expected_sha256", event.target.dataset.previewHash || "");
  const report = await post(`/api/panels/${panelId}/intake`, data);
  if (state.panel.id !== panelId) return;
  const warnings = [];
  if (report.provenance_conflict) warnings.push("将合成采集导入一个已标记为合成的面板。");
  if (report.conflict_count) warnings.push("已有相同角色且字节一致的采集，本批不会部分导入。");
  const rows = report.members.map((member) => ({
    title: member.title, role: state.catalog.roles[member.role], dimensions: member.dimensions,
    has_calibration: member.has_calibration, existing_title: member.existing_title,
  }));
  showReport($("packet-report"), {
    title: save ? "采集包写入成功" : "采集包预检",
    summary: save ? `已导入 ${report.imported.length} 张原图。` : "校验了每张原图、属性和对应标定。",
    metrics: {
      payload_images: report.members.length,
      importable_images: report.importable_count,
      synthetic: report.synthetic,
      source_panel: report.original_panel,
    },
    rows,
    warnings,
  });
  $("packet-import").elements.save.disabled = save || report.importable_count === 0;
  event.target.dataset.previewHash = save ? "" : report.sha256;
  if (save) {
    await refreshPanels();
    await refreshCaptures();
    displayPanel();
    notice("已完成采集包导入。");
  }
});

async function initialize() {
  const catalog = await json("/api/catalog");
  state.catalog = catalog;
  $("online").textContent = "本地资料库在线";
  const choices = Object.entries(catalog.roles);
  setOptions($("upload-role"), choices, "other");
  setOptions($("capture-filter"), choices, undefined, "所有角色");
  setOptions($("record-filter"), Object.entries(catalog.review_states), undefined, "所有状态");
  const themes = new Map();
  for (const tool of catalog.tools) {
    if (!themes.has(tool.group)) themes.set(tool.group, node("optgroup", null, {label: tool.group}));
    themes.get(tool.group).append(node("option", tool.title, {value: tool.key}));
  }
  $("tool").replaceChildren(...themes.values());
  $("tool").value = "rectify";
  await displayTool();
  await refreshPanels();
  const cached = localStorage.getItem("lumagrid.panel");
  const panel = state.panels.find((item) => item.id === cached);
  if (panel) await selectPanel(panel, localStorage.getItem("lumagrid.capture"));
}

initialize().catch((error) => notice(error.message, true));
