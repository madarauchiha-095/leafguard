const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const palette = ["#176b4a", "#609b74", "#9bb6a0", "#b18b49", "#51758a"];
let csrfToken = "";
let resultPage = 0;
const PAGE_SIZE = 100;

async function csrf() {
  const response = await fetch("/api/auth/csrf", { credentials: "same-origin" });
  const data = await response.json();
  csrfToken = data.csrf_token;
  return csrfToken;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== "GET") {
    if (!csrfToken) await csrf();
    headers.set("X-CSRF-Token", csrfToken);
  }
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  const data = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) throw new Error(data?.error || `Request failed (${response.status})`);
  return data;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function prettyClass(value) {
  return String(value || "").replace("___", " / ").replaceAll("_", " ").trim();
}

function showNotice(element, message, kind = "error") {
  if (!element) return;
  element.textContent = message;
  element.className = `notice ${kind}`;
}

async function bindAuthForm() {
  const form = $("[data-auth-form]");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(form));
    const message = $(".notice", form);
    const isRegister = form.dataset.authForm === "register";
    if (isRegister && values.password !== values.confirm) {
      showNotice(message, "Passwords do not match.");
      return;
    }
    try {
      const result = await api(isRegister ? "/api/auth/register" : "/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: values.username, password: values.password }),
      });
      showNotice(message, result.message || "Signed in.", "success");
      setTimeout(() => location.assign(isRegister ? "/login" : "/dashboard"), 450);
    } catch (error) {
      showNotice(message, error.message);
    }
  });
}

function bindLogout() {
  const button = $("[data-logout]");
  if (!button) return;
  button.addEventListener("click", async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
      location.assign("/");
    } catch (error) {
      alert(error.message);
    }
  });
}

function drawGroupedBars(container, rows, categories, series, label) {
  if (!rows.length) {
    container.innerHTML = '<div class="chart-empty">No values are available for this chart.</div>';
    return;
  }
  const width = 760, height = 285, left = 48, right = 12, top = 16, bottom = 72;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const max = Math.max(1, ...rows.flatMap((row) => series.map((item) => Number(row[item.key]) || 0)));
  const step = plotWidth / Math.max(1, categories.length);
  const groupWidth = step * 0.72;
  const barWidth = groupWidth / series.length;
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)} comparison">`;

  for (let tick = 0; tick <= 4; tick++) {
    const value = max * tick / 4;
    const y = top + plotHeight - plotHeight * tick / 4;
    svg += `<line x1="${left}" x2="${width - right}" y1="${y}" y2="${y}" stroke="#e7ece8"/>`;
    svg += `<text x="${left - 8}" y="${y + 4}" text-anchor="end" fill="#7a8980" font-size="10">${value.toFixed(1)}</text>`;
  }

  rows.forEach((row, categoryIndex) => {
    const start = left + categoryIndex * step + (step - groupWidth) / 2;
    series.forEach((item, seriesIndex) => {
      const value = Number(row[item.key]) || 0;
      const barHeight = plotHeight * value / max;
      const x = start + seriesIndex * barWidth + 1;
      const y = top + plotHeight - barHeight;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(2, barWidth - 3)}" height="${barHeight}" rx="3" fill="${item.color}"><title>${escapeHtml(categories[categoryIndex])} / ${escapeHtml(item.name)}: ${value.toFixed(1)}</title></rect>`;
    });
    const category = String(categories[categoryIndex]);
    const shortLabel = category.length > 20 ? `${category.slice(0, 19)}…` : category;
    svg += `<text x="${left + categoryIndex * step + step / 2}" y="${top + plotHeight + 18}" text-anchor="middle" fill="#59675f" font-size="10">${escapeHtml(shortLabel)}</text>`;
  });
  svg += "</svg>";
  const legend = series.map((item) => `<span class="legend-item"><i class="swatch" style="background:${item.color}"></i>${escapeHtml(item.name)}</span>`).join("");
  container.innerHTML = `${svg}<div class="chart-legend">${legend}</div>`;
}

function predictionRows(images) {
  return images.flatMap((image) => Object.entries(image.models || {}).map(([key, model]) => ({
    image, key, model: model.model, predicted: prettyClass(model.predicted_class),
    confidence: model.confidence, latency: model.latency_ms, top3: model.top3 || [],
  })));
}

function toCsv(rows) {
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const fields = ["filename", "model", "predicted_class", "confidence", "top_three", "latency_ms", "sha256"];
  return [fields.join(","), ...rows.map((row) => [
    row.image.filename, row.model, row.predicted, row.confidence,
    row.top3.map((item) => `${prettyClass(item.class)} (${(100 * item.confidence).toFixed(1)}%)`).join("; "),
    row.latency, row.image.sha256,
  ].map(quote).join(","))].join("\r\n");
}

function downloadFile(name, blob) {
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

function imageDimensions(file) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("File is not a readable image."));
    };
    image.src = objectUrl;
  });
}

async function sha256(file) {
  const bytes = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function setProgress(done, total, message) {
  $("#run-status").textContent = message;
  $("#progress-fill").style.width = `${total ? 100 * done / total : 0}%`;
}

function setupUpload() {
  const folderInput = $("#folder-input");
  const filesInput = $("#files-input");
  const dropzone = $("#dropzone");
  const runButton = $("#analyze-button");
  if (!folderInput || !filesInput) return;

  let selectedFiles = [];
  let completedImages = [];
  let isRunning = false;
  let isReady = false;
  let controller = null;
  const fileLimit = 10 * 1024 ** 2;
  const pixelLimit = 20_000_000;

  function setSelectedFiles(files) {
    selectedFiles = files;
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    $("#selection-summary").textContent = `${files.length.toLocaleString()} image(s) selected · ${(totalBytes / 1048576).toFixed(1)} MB total`;
    runButton.disabled = !files.length || !isReady;
    $("#selection-warning").textContent = "";
    $("#upload-list").innerHTML = files.slice(0, 6).map((file) =>
      `<div class="small">${escapeHtml(file.webkitRelativePath || file.name)} · ${(file.size / 1024).toFixed(0)} KB</div>`
    ).join("") + (files.length > 6 ? `<div class="small">and ${files.length - 6} more files</div>` : "");
  }

  folderInput.addEventListener("change", () => setSelectedFiles([...folderInput.files]));
  filesInput.addEventListener("change", () => setSelectedFiles([...filesInput.files]));
  $("#choose-folder").addEventListener("click", () => folderInput.click());
  $("#choose-files").addEventListener("click", () => filesInput.click());
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("drag"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
    setSelectedFiles([...event.dataTransfer.files]);
  });
  $("#batch-cancel").addEventListener("click", () => controller?.abort());
  $("#page-prev").addEventListener("click", () => {
    resultPage = Math.max(0, resultPage - 1);
    renderResults(completedImages);
  });
  $("#page-next").addEventListener("click", () => {
    resultPage += 1;
    renderResults(completedImages);
  });

  runButton.addEventListener("click", async () => {
    if (isRunning || !selectedFiles.length) return;
    isRunning = true;
    resultPage = 0;
    controller = new AbortController();
    $("#batch-cancel").classList.remove("hidden");
    runButton.disabled = true;
    completedImages = [];
    $("#analysis-results").classList.remove("hidden");
    const validFiles = [];
    const failedFiles = [];

    for (let index = 0; index < selectedFiles.length; index++) {
      const file = selectedFiles[index];
      if (file.size > fileLimit) {
        failedFiles.push({ filename: file.name, error: "File exceeds 10 MB." });
        continue;
      }
      try {
        const dimensions = await imageDimensions(file);
        if (dimensions.width * dimensions.height > pixelLimit) throw new Error("Image exceeds 20 megapixels.");
        const id = `${index}-${await sha256(file)}`;
        validFiles.push({ file, ...dimensions, id });
      } catch (error) {
        failedFiles.push({ filename: file.name, error: error.message || "Unsupported image." });
      }
    }

    const chunkSize = 8;
    let processed = 0;
    try {
      await csrf();
      if (!validFiles.length) {
        renderResults(completedImages, failedFiles);
        $("#run-status").textContent = "No supported images were ready to analyze.";
        return;
      }
      const run = await api("/api/analysis/runs", { method: "POST" });
      $("#saved-history-link").classList.remove("hidden");
      for (let offset = 0; offset < validFiles.length; offset += chunkSize) {
        const chunk = validFiles.slice(offset, offset + chunkSize);
        const form = new FormData();
        form.append("run_id", run.id);
        chunk.forEach((item) => {
          form.append("files", item.file, item.file.webkitRelativePath || item.file.name);
          form.append("client_ids", item.id);
        });
        const response = await api("/api/analyze", { method: "POST", body: form, signal: controller.signal });
        response.results.forEach((result, index) => {
          const input = chunk.find((item) => item.id === result.client_id) || chunk[index];
          if (result.error) failedFiles.push({ filename: result.filename, error: result.error });
          else completedImages.push({ ...result, file: input.file, inputId: input.id });
        });
        processed += chunk.length;
        setProgress(processed, validFiles.length, `Analyzed ${processed} of ${validFiles.length} images`);
      }
      renderResults(completedImages, failedFiles);
      $("#run-status").textContent = `Completed · ${completedImages.length.toLocaleString()} analyzed · ${failedFiles.length.toLocaleString()} skipped · results saved`;
      if (completedImages.length) showInspector(completedImages[0]);
    } catch (error) {
      $("#run-status").textContent = controller.signal.aborted
        ? "Analysis cancelled. Completed results are saved to your dashboard."
        : `Analysis stopped: ${error.message}`;
      renderResults(completedImages, failedFiles);
    } finally {
      isRunning = false;
      controller = null;
      $("#batch-cancel").classList.add("hidden");
      runButton.disabled = !selectedFiles.length;
    }
  });

  $("#model-filter").addEventListener("change", () => {
    resultPage = 0;
    renderResults(completedImages);
  });
  return () => {
    isReady = true;
    runButton.disabled = !selectedFiles.length;
  };
}

function renderResults(images, failures = []) {
  const rows = predictionRows(images);
  $("#result-count").textContent = `${rows.length.toLocaleString()} predictions`;
  const unanimous = images.filter((image) => new Set(Object.values(image.models).map((model) => model.predicted_class)).size === 1).length;
  const meanScore = rows.length ? (100 * rows.reduce((sum, row) => sum + row.confidence, 0) / rows.length).toFixed(1) : "—";
  $("#result-summary").innerHTML = `
    <div class="stat-card"><div class="label">IMAGES ANALYZED</div><div class="value">${images.length.toLocaleString()}</div></div>
    <div class="stat-card"><div class="label">MODEL PREDICTIONS</div><div class="value">${rows.length.toLocaleString()}</div></div>
    <div class="stat-card"><div class="label">FOUR-MODEL AGREEMENT</div><div class="value">${unanimous.toLocaleString()}</div></div>
    <div class="stat-card"><div class="label">MEAN TOP-1 SCORE</div><div class="value">${meanScore}${rows.length ? "%" : ""}</div></div>`;

  const counts = {};
  rows.forEach((row) => {
    const key = `${row.key}|${row.predicted}`;
    counts[key] = (counts[key] || 0) + 1;
  });
  const categories = [...new Set(rows.map((row) => row.predicted))];
  const chartRows = categories.map((category) => {
    const result = { category };
    Object.keys(window.LG_MODELS).forEach((key) => { result[key] = counts[`${key}|${category}`] || 0; });
    return result;
  });
  drawGroupedBars(
    $("#prediction-chart"), chartRows, categories,
    Object.entries(window.LG_MODELS).map(([key, name], index) => ({ key, name, color: palette[index] })),
    "Image count",
  );

  const modelFilter = $("#model-filter").value;
  const visibleRows = modelFilter === "all" ? rows : rows.filter((row) => row.key === modelFilter);
  const pageCount = Math.max(1, Math.ceil(visibleRows.length / PAGE_SIZE));
  resultPage = Math.min(resultPage, pageCount - 1);
  const pageRows = visibleRows.slice(resultPage * PAGE_SIZE, (resultPage + 1) * PAGE_SIZE);
  $("#page-label").textContent = visibleRows.length
    ? `Rows ${resultPage * PAGE_SIZE + 1}-${Math.min((resultPage + 1) * PAGE_SIZE, visibleRows.length)} of ${visibleRows.length.toLocaleString()}`
    : "No predictions";
  $("#page-prev").disabled = resultPage === 0;
  $("#page-next").disabled = resultPage >= pageCount - 1;
  $("#result-body").innerHTML = pageRows.map((row) => `
    <tr><td><button class="text-button" data-inspect="${escapeHtml(row.image.client_id)}">${escapeHtml(row.image.filename)}</button></td>
    <td>${escapeHtml(row.model)}</td><td>${escapeHtml(row.predicted)}</td>
    <td><span class="bar"><i style="width:${Math.max(0, Math.min(100, row.confidence * 100))}%"></i></span><span class="score">${(row.confidence * 100).toFixed(1)}%</span></td>
    <td><details><summary>Top 3</summary>${row.top3.map((item) => `<div class="small">${escapeHtml(prettyClass(item.class))} · ${(100 * item.confidence).toFixed(1)}%</div>`).join("")}</details></td>
    <td>${row.latency.toFixed(1)}</td></tr>`).join("");
  $$("[data-inspect]").forEach((button) => button.addEventListener("click", () => {
    const image = images.find((item) => item.client_id === button.dataset.inspect);
    if (image) showInspector(image);
  }));

  $("#export-csv").onclick = () => downloadFile("leafguard-predictions.csv", new Blob([toCsv(rows)], { type: "text/csv;charset=utf-8" }));
  $("#export-json").onclick = () => downloadFile("leafguard-predictions.json", new Blob([
    JSON.stringify(images.map(({ file, ...image }) => image), null, 2),
  ], { type: "application/json" }));
  $("#result-table-wrap").classList.toggle("hidden", !rows.length);
  $("#failed-list").innerHTML = failures.length
    ? `<details class="card"><summary>${failures.length} file(s) skipped</summary>${failures.slice(0, 50).map((item) => `<p class="small">${escapeHtml(item.filename)} — ${escapeHtml(item.error)}</p>`).join("")}${failures.length > 50 ? `<p class="small">Showing the first 50 skipped files.</p>` : ""}</details>`
    : "";
  $("#analysis-results").classList.remove("hidden");
}

function showInspector(row) {
  $("#inspect-title").textContent = row.filename;
  $("#inspect-input").src = URL.createObjectURL(row.file);
  const modelSelect = $("#explain-model");
  modelSelect.innerHTML = Object.entries(window.LG_MODELS).map(([key, name]) =>
    `<option value="${key}">${escapeHtml(name)}</option>`
  ).join("");
  $("#explain-button").onclick = async () => {
    const button = $("#explain-button");
    button.disabled = true;
    button.textContent = "Computing explanation…";
    try {
      const form = new FormData();
      form.append("file", row.file, row.file.name);
      form.append("model", modelSelect.value);
      const result = await api("/api/explain", { method: "POST", body: form });
      const figures = [
        ["Input", result.original], ["Grad-CAM++", result.overlay],
        ...Object.entries(result.branches).map(([key, value]) => [`${key} branch`, value]),
      ];
      $("#explanation-grid").innerHTML = figures.map(([name, data]) => `
        <figure><img src="data:image/png;base64,${data}" alt="${escapeHtml(name)}"><figcaption>${escapeHtml(name)}</figcaption></figure>`).join("");
      $("#explain-summary").textContent = `${prettyClass(result.class_name)} · score ${(result.confidence * 100).toFixed(1)}% · ${result.interpretation}`;
    } catch (error) {
      $("#explain-summary").textContent = error.message;
    } finally {
      button.disabled = false;
      button.textContent = "Generate Grad-CAM++";
    }
  };
}

async function initWorkspace() {
  const enableAnalysis = setupUpload();
  try {
    const user = await api("/api/auth/me");
    $("#user-name").textContent = user.username;
    csrfToken = user.csrf_token;
    window.LG_MODELS = await api("/api/models");
    $("#model-filter").innerHTML = '<option value="all">All models</option>' + Object.entries(window.LG_MODELS)
      .map(([key, name]) => `<option value="${key}">${escapeHtml(name)}</option>`).join("");
    enableAnalysis();
  } catch {
    location.assign("/login");
    return;
  }
}

function formatDate(value) {
  const date = new Date(`${value.replace(" ", "T")}Z`);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], {
    dateStyle: "medium", timeStyle: "short",
  });
}

function historyAgreement(run) {
  return run.images_analyzed
    ? `${(100 * run.agreement_count / run.images_analyzed).toFixed(1)}%`
    : "—";
}

function historyConfidence(run) {
  return run.prediction_count
    ? `${(100 * run.confidence_sum / run.prediction_count).toFixed(1)}%`
    : "—";
}

async function loadHistory() {
  const data = await api("/api/analysis/history");
  const summary = data.summary;
  $("#history-run-count").textContent = Number(summary.total_runs).toLocaleString();
  $("#history-image-count").textContent = Number(summary.images_analyzed).toLocaleString();
  $("#history-agreement").textContent = summary.images_analyzed
    ? `${(100 * summary.agreement_count / summary.images_analyzed).toFixed(1)}%`
    : "—";
  $("#history-confidence").textContent = summary.prediction_count
    ? `${(100 * summary.confidence_sum / summary.prediction_count).toFixed(1)}%`
    : "—";
  const body = $("#history-body");
  body.innerHTML = data.runs.map((run) => `
    <tr>
      <td><strong>Saved analysis</strong><div class="small">${escapeHtml(run.id.slice(0, 8))}</div></td>
      <td>${escapeHtml(formatDate(run.created_at))}</td>
      <td>${Number(run.images_analyzed).toLocaleString()}</td>
      <td>${historyAgreement(run)}</td>
      <td>${historyConfidence(run)}</td>
      <td><div class="actions">
        <button class="btn btn-secondary" data-history-view="${escapeHtml(run.id)}">View results</button>
        <button class="btn btn-quiet" data-history-delete="${escapeHtml(run.id)}" aria-label="Delete saved analysis">Delete</button>
      </div></td>
    </tr>`).join("");
  const isEmpty = data.runs.length === 0;
  $("#history-empty").classList.toggle("hidden", !isEmpty);
  $("#history-table").classList.toggle("hidden", isEmpty);
  $("#history-status").textContent = isEmpty
    ? "No saved analyses yet."
    : `Showing ${data.runs.length} most recent ${data.runs.length === 1 ? "analysis" : "analyses"}.`;
  $$("[data-history-view]").forEach((button) => button.addEventListener("click", () => {
    historyPage = 0;
    loadHistoryDetail(button.dataset.historyView);
  }));
  $$("[data-history-delete]").forEach((button) => button.addEventListener("click", async () => {
    if (!confirm("Delete this saved analysis and its prediction history?")) return;
    button.disabled = true;
    try {
      await api(`/api/analysis/history/${encodeURIComponent(button.dataset.historyDelete)}/delete`, { method: "POST" });
      $("#history-detail").classList.add("hidden");
      await loadHistory();
    } catch (error) {
      button.disabled = false;
      $("#history-status").textContent = error.message;
    }
  }));
}

let historyPage = 0;
let activeHistoryRun = null;

async function loadHistoryDetail(runId) {
  activeHistoryRun = runId;
  try {
    const data = await api(`/api/analysis/history/${encodeURIComponent(runId)}?page=${historyPage}`);
    const run = data.run;
    $("#history-detail-title").textContent = "Saved analysis";
    $("#history-detail-subtitle").textContent =
      `${formatDate(run.created_at)} · ${Number(run.images_analyzed).toLocaleString()} images · ${historyAgreement(run)} four-model agreement`;
    const rows = predictionRows(data.results);
    $("#history-detail-body").innerHTML = rows.map((row) => `
      <tr><td>${escapeHtml(row.image.filename)}</td><td>${escapeHtml(row.model)}</td>
      <td>${escapeHtml(row.predicted)}</td><td>${(100 * row.confidence).toFixed(1)}%</td>
      <td>${row.top3.map((item) => `${escapeHtml(prettyClass(item.class))} · ${(100 * item.confidence).toFixed(1)}%`).join("<br>")}</td>
      <td>${Number(row.latency).toFixed(1)} ms</td></tr>`).join("");
    const pages = Math.max(1, Math.ceil(data.total / data.page_size));
    $("#history-page-label").textContent = data.total
      ? `Images ${data.page * data.page_size + 1}–${Math.min((data.page + 1) * data.page_size, data.total)} of ${Number(data.total).toLocaleString()}`
      : "No saved predictions";
    $("#history-page-prev").disabled = data.page === 0;
    $("#history-page-next").disabled = data.page >= pages - 1;
    $("#history-detail").classList.remove("hidden");
    $("#history-export-csv").onclick = () => {
      const csvRows = rows.map((row) => ({
        image: row.image, model: row.model, predicted: row.predicted,
        confidence: row.confidence, top3: row.top3, latency: row.latency,
      }));
      downloadFile("leafguard-saved-predictions-page.csv", new Blob([toCsv(csvRows)], { type: "text/csv;charset=utf-8" }));
    };
    $("#history-export-json").onclick = () => downloadFile(
      "leafguard-saved-predictions-page.json",
      new Blob([JSON.stringify(data.results, null, 2)], { type: "application/json" }),
    );
    $("#history-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    $("#history-status").textContent = error.message;
  }
}

async function initDashboard() {
  try {
    const user = await api("/api/auth/me");
    csrfToken = user.csrf_token;
    $("#user-name").textContent = user.username;
    $("#welcome-name").textContent = `, ${user.username}`;
    await loadHistory();
    $("#history-page-prev").addEventListener("click", () => {
      if (historyPage > 0) {
        historyPage -= 1;
        loadHistoryDetail(activeHistoryRun);
      }
    });
    $("#history-page-next").addEventListener("click", () => {
      historyPage += 1;
      loadHistoryDetail(activeHistoryRun);
    });
  } catch (error) {
    if (error.message.includes("Sign-in required") || error.message.includes("401")) location.assign("/login");
    else $("#history-status").textContent = error.message;
  }
}

async function initInsights() {
  try {
    const data = await api("/api/research");
    const keys = Object.keys(data.models);
    const metrics = keys.map((key) => {
      const model = data.models[key];
      return {
        model: model.name,
        accuracy: { pv: model.plantvillage.accuracy, pd: model.plantdoc.accuracy, ps: model.plantseg_v7.accuracy },
        f1: { pv: model.plantvillage.f1_macro, pd: model.plantdoc.f1_macro, ps: model.plantseg_v7.f1_macro },
      };
    });
    const datasets = [
      { key: "pv", name: "PlantVillage", color: palette[0] },
      { key: "pd", name: "PlantDoc", color: palette[1] },
      { key: "ps", name: "PlantSeg v7", color: palette[3] },
    ];
    drawGroupedBars($("#accuracy-chart"), metrics.map((item) => ({ category: item.model, ...item.accuracy })), metrics.map((item) => item.model), datasets, "Accuracy");
    drawGroupedBars($("#f1-chart"), metrics.map((item) => ({ category: item.model, ...item.f1 })), metrics.map((item) => item.model), datasets, "Macro F1");
    $("#metric-body").innerHTML = keys.map((key) => {
      const model = data.models[key];
      return `<tr><td>${escapeHtml(model.name)}</td><td>${(model.plantvillage.accuracy * 100).toFixed(2)}%</td>
        <td>${(model.plantdoc.accuracy * 100).toFixed(2)}%</td><td>${(model.plantdoc.f1_macro * 100).toFixed(2)}%</td>
        <td>${(model.plantseg_v7.accuracy * 100).toFixed(2)}%</td><td>${(model.plantseg_v7.f1_macro * 100).toFixed(2)}%</td>
        <td>${Number(model.resources?.parameter_count || 0).toLocaleString()}</td>
        <td>${Number(model.resources?.checkpoint_size_mib || 0).toFixed(2)}</td>
        <td>${Number(model.resources?.inference_latency_ms_batch1 || 0).toFixed(1)}</td></tr>`;
    }).join("");
    $("#limitations").innerHTML = data.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    $("#research-status").textContent = "Metrics loaded from project evaluation artifacts.";
  } catch (error) {
    $("#research-status").textContent = error.message;
  }
}

async function initSecurity() {
  try {
    const data = await api("/api/security");
    $("#integrity-status").innerHTML = `<span class="status ${data.all_verified ? "" : "fail"}">${data.all_verified ? "All checkpoints verified" : "Integrity check failed"}</span>`;
    $("#device-name").textContent = data.device;
    $("#security-controls").innerHTML = data.controls.map((control) => `
      <tr><td>${escapeHtml(control.name)}</td><td><span class="tag">${escapeHtml(control.status)}</span></td><td>${escapeHtml(control.detail)}</td></tr>`).join("");
    $("#checkpoint-rows").innerHTML = Object.values(data.checkpoints).map((checkpoint) => `
      <tr><td>${escapeHtml(checkpoint.model)}</td><td>${escapeHtml(checkpoint.path)}</td>
      <td>${checkpoint.verified ? '<span class="status">Verified</span>' : '<span class="status fail">Mismatch</span>'}</td>
      <td><code>${escapeHtml(checkpoint.sha256 || "missing")}</code></td></tr>`).join("");
  } catch (error) {
    $("#security-controls").textContent = error.message;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await bindAuthForm();
  bindLogout();
  if (document.body.dataset.page === "dashboard") await initDashboard();
  if (document.body.dataset.page === "workspace") await initWorkspace();
  if (document.body.dataset.page === "insights") await initInsights();
  if (document.body.dataset.page === "security") await initSecurity();
});
