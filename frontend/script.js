let currentFile = null;
let currentImageDataUrl = null;
let cameraStream = null;
let cameraModal = null;
let analysisHistory = [];
let deepAnalysisData = [];
let deepAnalysisMeta = null;
let canvasVisible = true;
let deepSectionInitialHtml = "";
let activeImageToken = 0;
let projectModal = null;
let projectList = [];
let currentProjectId = null;
let streamSessionId = null;
let streamSummarySessionId = null;
let streamRunning = false;
let streamLoopTimer = null;
let streamRequestBusy = false;
let streamFrameIndex = 0;
let streamStartTsMs = 0;
let streamProcessedFrames = 0;
let streamCandidateFrames = 0;
let streamCandidateStore = {};
let currentInputSource = "upload";

const API_DEEP = "/api/analyze/deep";
const API_PROJECTS = "/api/projects";
const API_HISTORY = "/api/history";
const API_STREAM_START = "/api/stream/session/start";
const API_STREAM_FRAME = "/api/stream/frame/basic";
const API_STREAM_SUMMARY = (sessionId) => `/api/stream/session/${sessionId}/summary`;
const API_STREAM_REPORT = (sessionId) => `/api/stream/session/${sessionId}/report`;
const API_STREAM_REPORT_BUNDLE = (sessionId) => `/api/stream/session/${sessionId}/report/bundle`;
const API_STREAM_DELETE = (sessionId) => `/api/stream/session/${sessionId}`;
const DEFAULT_BASIC_CONF = 0.25;
const DEFAULT_DEEP_CONF = 0.25;
const DEFAULT_IOU = 0.6;
const DEFAULT_BASIC_IMGSZ = 640;
const DEFAULT_DEEP_IMGSZ = 640;
const DEFAULT_DEVICE = "auto";
const CAMERA_BASIC_CONF = 0.25;
const CAMERA_DEEP_CONF = 0.25;
const CAMERA_BASIC_IMGSZ = 640;
const CAMERA_DEEP_IMGSZ = 640;
const CAMERA_IOU = 0.6;
const CAMERA_CAPTURE_MIME = "image/png";
const CAMERA_CAPTURE_JPEG_QUALITY = 0.95;
const STREAM_SEND_INTERVAL_MS = 260;
const STREAM_FRAME_JPEG_QUALITY = 0.72;
const STREAM_MAX_CANDIDATE_CACHE = 200;

async function waitForStreamIdle(maxWaitMs = 2500) {
    const deadline = Date.now() + Math.max(100, Number(maxWaitMs) || 2500);
    while (streamRequestBusy && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
}

function normalizeConfidence01(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1, n));
}

document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("cameraModal")) {
        cameraModal = new bootstrap.Modal(document.getElementById("cameraModal"));
        document.getElementById("cameraModal").addEventListener("hidden.bs.modal", async () => {
            await stopRealtimeScan(false);
            stopCameraStream();
            setStreamButtonsState();
            setStreamLiveStatus("San sang stream realtime 5 lop");
        });
    }
    if (document.getElementById("projectModal")) {
        projectModal = new bootstrap.Modal(document.getElementById("projectModal"));
    }

    document.getElementById("fileInput").addEventListener("change", handleFileSelect);
    deepSectionInitialHtml = document.getElementById("deepAnalysisSection").innerHTML;
    loadProjects().then(() => loadHistory()).then(renderHistory).catch(() => renderHistory());
    setStreamButtonsState();
    setStreamLiveStatus("San sang stream realtime 5 lop");
    window.addEventListener("resize", () => {
        const img = document.getElementById("analysisImage");
        if (img && img.src && img.complete && (img.naturalWidth || img.width)) {
            setupCanvas();
        }
    });
});

function newImageToken() {
    activeImageToken = Date.now() + Math.floor(Math.random() * 1000);
    return activeImageToken;
}

function resetDeepAnalysisUI(showActionButton = false) {
    const deepSection = document.getElementById("deepAnalysisSection");
    deepAnalysisData = [];
    deepAnalysisMeta = null;
    canvasVisible = true;
    document.getElementById("detectionCanvas").style.opacity = "1";

    if (deepSectionInitialHtml) {
        deepSection.innerHTML = deepSectionInitialHtml;
    }
    deepSection.style.display = showActionButton ? "block" : "none";
}

function setProgressVisible(show) {
    document.getElementById("analysisProgress").style.display = show ? "block" : "none";
}

function updateProgress(percent, status) {
    document.getElementById("progressBarFill").style.width = `${percent}%`;
    document.getElementById("progressPercentage").textContent = `${percent}%`;
    document.getElementById("progressStatus").textContent = status;
}

function setStreamLiveStatus(text, cssClass = "") {
    const el = document.getElementById("streamLiveStatus");
    if (!el) return;
    el.className = `stream-live-status ${cssClass}`.trim();
    el.textContent = text;
}

function setStreamButtonsState() {
    const startBtn = document.getElementById("streamStartBtn");
    const stopBtn = document.getElementById("streamStopBtn");
    const captureBtn = document.getElementById("captureImageBtn");
    if (startBtn) startBtn.disabled = streamRunning;
    if (stopBtn) stopBtn.disabled = !streamRunning;
    if (captureBtn) captureBtn.disabled = streamRunning;
}

async function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function dataURLtoFile(dataUrl, filename = "capture.jpg") {
    const arr = dataUrl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
}

async function callAnalyzeApi(endpoint, file, options = {}) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("conf", String(options.conf ?? DEFAULT_BASIC_CONF));
    formData.append("iou", String(options.iou ?? DEFAULT_IOU));
    formData.append("imgsz", String(options.imgsz ?? DEFAULT_BASIC_IMGSZ));
    formData.append("device", String(options.device ?? DEFAULT_DEVICE));
    formData.append("input_source", String(options.inputSource ?? currentInputSource ?? "upload"));
    formData.append("scene", String(options.scene ?? "default"));
    formData.append("project_id", String(currentProjectId || 0));

    const res = await fetch(endpoint, { method: "POST", body: formData });
    if (!res.ok) {
        let message = `API error: ${res.status}`;
        try {
            const err = await res.json();
            message = err.detail || message;
        } catch (_) {}
        throw new Error(message);
    }
    return await res.json();
}

async function callJsonApi(endpoint, method = "GET", body = null) {
    const init = { method, headers: {} };
    if (body !== null) {
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
    }
    const res = await fetch(endpoint, init);
    if (!res.ok) {
        let message = `API error: ${res.status}`;
        try {
            const err = await res.json();
            message = err.detail || message;
        } catch (_) {}
        throw new Error(message);
    }
    return await res.json();
}

function getStoredProjectId() {
    const raw = localStorage.getItem("activeProjectId");
    const n = Number(raw);
    if (!Number.isFinite(n) || n <= 0) return null;
    return n;
}

function setStoredProjectId(projectId) {
    const n = Number(projectId);
    if (Number.isFinite(n) && n > 0) {
        localStorage.setItem("activeProjectId", String(n));
    } else {
        localStorage.removeItem("activeProjectId");
    }
}

function renderProjectSelect() {
    const select = document.getElementById("projectSelect");
    if (!select) return;
    if (!projectList.length) {
        select.innerHTML = `<option value="">Chua co du an</option>`;
        return;
    }
    select.innerHTML = projectList
        .map((p) => `<option value="${p.id}">${p.name} (${p.history_count || 0})</option>`)
        .join("");
    if (!currentProjectId) currentProjectId = projectList[0].id;
    select.value = String(currentProjectId);
}

async function loadProjects() {
    const out = await callJsonApi(`${API_PROJECTS}?limit=200`);
    projectList = Array.isArray(out.items) ? out.items : [];
    const preferred = getStoredProjectId();
    if (preferred && projectList.some((p) => Number(p.id) === preferred)) {
        currentProjectId = preferred;
    } else if (out.active_project_id) {
        currentProjectId = Number(out.active_project_id);
    } else if (projectList.length) {
        currentProjectId = Number(projectList[0].id);
    } else {
        currentProjectId = null;
    }
    setStoredProjectId(currentProjectId);
    renderProjectSelect();
}

async function refreshProjects() {
    try {
        await loadProjects();
        await loadHistory();
        renderHistory();
    } catch (err) {
        alert(err.message || "Khong the tai danh sach du an.");
    }
}

async function onProjectChange(projectId) {
    const n = Number(projectId);
    currentProjectId = Number.isFinite(n) && n > 0 ? n : null;
    setStoredProjectId(currentProjectId);
    try {
        await loadHistory();
        renderHistory();
    } catch (_) {
        renderHistory();
    }
}

function openProjectModal() {
    if (!projectModal) return;
    document.getElementById("projectNameInput").value = "";
    document.getElementById("projectDescInput").value = "";
    projectModal.show();
}

async function createProjectFromModal() {
    const nameEl = document.getElementById("projectNameInput");
    const descEl = document.getElementById("projectDescInput");
    const name = (nameEl?.value || "").trim();
    const description = (descEl?.value || "").trim();
    if (!name) {
        alert("Vui long nhap ten du an.");
        return;
    }
    try {
        const out = await callJsonApi(API_PROJECTS, "POST", { name, description });
        const pid = Number(out?.project?.id || 0);
        await loadProjects();
        if (pid > 0) {
            currentProjectId = pid;
            setStoredProjectId(pid);
            renderProjectSelect();
        }
        await loadHistory();
        renderHistory();
        if (projectModal) projectModal.hide();
    } catch (err) {
        alert(err.message || "Khong tao duoc du an.");
    }
}

async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file || !file.type.startsWith("image/")) return;

    currentInputSource = "upload";
    currentFile = file;
    currentImageDataUrl = await fileToDataUrl(file);
    const token = newImageToken();
    showAnalysisSection(currentImageDataUrl);
    await performBasicAnalysis(token, {
        conf: DEFAULT_DEEP_CONF,
        iou: DEFAULT_IOU,
        imgsz: DEFAULT_DEEP_IMGSZ,
    });
    event.target.value = "";
}

function showAnalysisSection(imageSrc) {
    document.getElementById("uploadSection").style.display = "none";
    document.getElementById("analysisSection").style.display = "block";
    document.getElementById("basicResultCard").style.display = "none";
    resetDeepAnalysisUI(false);

    const img = document.getElementById("analysisImage");
    img.src = imageSrc;
    img.onload = setupCanvas;
    clearCanvas();
}

function setupCanvas() {
    const img = document.getElementById("analysisImage");
    const container = document.getElementById("imageContainer");
    const canvas = document.getElementById("detectionCanvas");
    const naturalW = img.naturalWidth || img.width || 1;
    const naturalH = img.naturalHeight || img.height || 1;
    const displayW = img.clientWidth || img.offsetWidth || naturalW;
    const displayH = img.clientHeight || img.offsetHeight || naturalH;

    canvas.width = naturalW;
    canvas.height = naturalH;
    canvas.style.width = `${displayW}px`;
    canvas.style.height = `${displayH}px`;
    canvas.style.left = `${img.offsetLeft}px`;
    canvas.style.top = `${img.offsetTop}px`;

    if (container) {
        canvas.style.maxWidth = `${container.clientWidth}px`;
    }
}

function clearCanvas() {
    const canvas = document.getElementById("detectionCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function setupStreamOverlay() {
    const video = document.getElementById("cameraStream");
    const canvas = document.getElementById("streamOverlayCanvas");
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.style.width = `${video.clientWidth}px`;
    canvas.style.height = `${video.clientHeight}px`;
}

function clearStreamOverlay() {
    const canvas = document.getElementById("streamOverlayCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawStreamOverlay(detections = [], meta = {}) {
    const video = document.getElementById("cameraStream");
    const canvas = document.getElementById("streamOverlayCanvas");
    if (!video || !canvas) return;
    setupStreamOverlay();
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections.length) return;

    ctx.lineWidth = 2;
    ctx.font = "bold 13px Arial";
    detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        const w = Math.max(1, x2 - x1);
        const h = Math.max(1, y2 - y1);
        const tid = det.track_id != null ? `#${det.track_id}` : "";
        const conf = normalizeConfidence01(det.confidence || 0) * 100;
        const label = `${tid} ${conf.toFixed(1)}%`.trim();

        ctx.strokeStyle = "rgba(239,68,68,0.95)";
        ctx.fillStyle = "rgba(239,68,68,0.95)";
        ctx.strokeRect(x1, y1, w, h);
        const tw = ctx.measureText(label).width + 10;
        const by = Math.max(0, y1 - 22);
        ctx.fillRect(x1, by, tw, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 + 5, by + 14);
    });

    if (meta.cached) {
        ctx.fillStyle = "rgba(245,158,11,0.95)";
        ctx.font = "bold 12px Arial";
        ctx.fillText("cached", 10, 18);
    }
}

function drawDetections(detections, colorForClass, width = 3) {
    const canvas = document.getElementById("detectionCanvas");
    const ctx = canvas.getContext("2d");
    clearCanvas();

    ctx.lineWidth = width;
    ctx.font = "bold 14px Arial";

    detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        const color = colorForClass(det.class_id);
        const conf = normalizeConfidence01(det.confidence);
        const label = `${formatClassName(det.class_name)} (${(conf * 100).toFixed(1)}%)`;

        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const textWidth = ctx.measureText(label).width;
        const boxY = Math.max(0, y1 - 24);
        ctx.fillRect(x1, boxY, textWidth + 10, 22);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x1 + 5, boxY + 15);
    });
}

async function performBasicAnalysis(imageToken = activeImageToken, inferOptions = {}) {
    if (!currentFile) return;
    setProgressVisible(true);
    updateProgress(5, "Khoi tao he thong...");

    try {
        updateProgress(20, "Dang gui anh len server...");
        const deep = await callAnalyzeApi(API_DEEP, currentFile, {
            conf: inferOptions.conf ?? DEFAULT_DEEP_CONF,
            iou: inferOptions.iou ?? DEFAULT_IOU,
            imgsz: inferOptions.imgsz ?? DEFAULT_DEEP_IMGSZ,
            device: DEFAULT_DEVICE,
            inputSource: currentInputSource,
            scene: inferOptions.scene ?? "default",
        });
        if (imageToken !== activeImageToken) return;
        updateProgress(80, "Dang xu ly ket qua...");

        const detections = deep.detections || [];
        const hasCrack = detections.length > 0;
        const maxConf = detections.length
            ? Math.max(...detections.map((d) => normalizeConfidence01(d.confidence))) * 100
            : 0;
        const confidence = hasCrack ? maxConf : 0;

        showBasicResult(hasCrack, confidence, deep);
        deepAnalysisData = detections;
        deepAnalysisMeta = deep;

        if (hasCrack) {
            drawDetections(detections, (cls) => classColor(cls), 3);
            renderDeepResultCards(detections, deep);
        } else {
            clearCanvas();
            renderDeepResultCards([], deep);
        }

        await loadHistory();
        renderHistory();
        updateProgress(100, "Hoan thanh.");
    } catch (err) {
        if (imageToken !== activeImageToken) return;
        clearCanvas();
        deepAnalysisData = [];
        deepAnalysisMeta = null;
        showError(err.message || "Phan tich 5 lop that bai");
    } finally {
        setTimeout(() => setProgressVisible(false), 300);
    }
}

function showBasicResult(hasCrack, confidence, basicMeta = null) {
    const resultCard = document.getElementById("basicResultCard");
    const statusIcon = document.getElementById("statusIcon");
    const statusTitle = document.getElementById("statusTitle");
    const statusDescription = document.getElementById("statusDescription");
    const confidenceValue = document.getElementById("confidenceValue");
    const deepAnalysisSection = document.getElementById("deepAnalysisSection");

    resultCard.style.display = "block";
    confidenceValue.textContent = confidence > 0 ? `${confidence.toFixed(1)}%` : "--";

    if (hasCrack) {
        resetDeepAnalysisUI(false);
        statusIcon.className = "status-icon has-crack";
        statusTitle.textContent = "Phat hien vet nut";
        statusTitle.style.color = "var(--danger-color)";
        statusDescription.textContent = "Da phat hien vet nut theo mo hinh RT-DETR 5 lop.";
    } else {
        resetDeepAnalysisUI(false);
        statusIcon.className = "status-icon no-crack";
        statusTitle.textContent = "Khong phat hien vet nut";
        statusTitle.style.color = "var(--success-color)";
        statusDescription.textContent = "Khong co box hop le tu mo hinh 5 lop.";
    }

    const t = Number(basicMeta?.infer_time_ms || 0);
    if (t > 0) {
        statusDescription.textContent += ` [${t.toFixed(0)} ms]`;
    }
    deepAnalysisSection.style.display = "block";
}

function showError(message) {
    const resultCard = document.getElementById("basicResultCard");
    const statusIcon = document.getElementById("statusIcon");
    const statusTitle = document.getElementById("statusTitle");
    const statusDescription = document.getElementById("statusDescription");
    const confidenceValue = document.getElementById("confidenceValue");
    const deepAnalysisSection = document.getElementById("deepAnalysisSection");

    resultCard.style.display = "block";
    statusIcon.className = "status-icon has-crack";
    statusTitle.textContent = "Loi phan tich";
    statusTitle.style.color = "var(--danger-color)";
    statusDescription.textContent = message;
    confidenceValue.textContent = "--";
    deepAnalysisSection.style.display = "none";
}

function classColor(classId) {
    const colors = ["#ef4444", "#f59e0b", "#eab308", "#84cc16", "#8b5cf6", "#06b6d4"];
    const idx = Math.abs(Number(classId)) % colors.length;
    return colors[idx];
}

function formatClassName(name) {
    const key = String(name).toLowerCase().replaceAll("_", " ").trim();
    const map = {
        crack: "vet nut",
        "longitudinal crack": "vet nut doc",
        "transverse crack": "vet nut ngang",
        "alligator crack": "nut da ca sau",
        "other corruption": "hu hong khac",
        pothole: "o ga",
        "crack unclassified": "vet nut chua phan loai",
        "vet nut chua phan loai": "vet nut chua phan loai",
    };
    return map[key] || String(name).replaceAll("_", " ");
}

async function performDeepAnalysis() {
    if (!currentFile) return;
    const deepConf = currentInputSource === "camera" ? CAMERA_DEEP_CONF : DEFAULT_DEEP_CONF;
    const deepIou = currentInputSource === "camera" ? CAMERA_IOU : DEFAULT_IOU;
    const deepImgsz = currentInputSource === "camera" ? CAMERA_DEEP_IMGSZ : DEFAULT_DEEP_IMGSZ;
    await performBasicAnalysis(activeImageToken, {
        conf: deepConf,
        iou: deepIou,
        imgsz: deepImgsz,
    });
}

function renderDeepResultCards(detections, meta = null) {
    const deepSection = document.getElementById("deepAnalysisSection");
    if (!detections.length) {
        deepSection.innerHTML = `<div class="alert alert-warning">Khong co box hop le o che do 5 lop.</div>`;
        return;
    }

    const grouped = {};
    detections.forEach((d) => {
        const key = `${d.class_id}|${d.class_name}`;
        if (!grouped[key]) grouped[key] = [];
        grouped[key].push(d);
    });

    const cards = Object.entries(grouped)
        .map(([key, items]) => {
            const [classIdStr, className] = key.split("|");
            const classId = Number(classIdStr);
            const avgConf = items.reduce((s, x) => s + normalizeConfidence01(x.confidence), 0) / items.length;
            const color = classColor(classId);
            const displayName = formatClassName(className);
            return `
                <div class="col-md-6 mb-3">
                    <div class="detection-summary-card" onclick="filterDetections(${classId})"
                         style="background: rgba(30,58,95,0.3); border-left:4px solid ${color}; padding:1rem; border-radius:8px; cursor:pointer;">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong style="color:${color};">${displayName}</strong>
                            <span class="badge" style="background:${color};">${items.length} vi tri</span>
                        </div>
                        <div style="color:rgba(255,255,255,0.75); margin-top:0.5rem;">
                            Do tin cay TB: ${(avgConf * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            `;
        })
        .join("");

    const severityInfo = meta
        ? `<div style="color:rgba(255,255,255,0.88); font-size:0.95rem; margin-bottom:0.6rem;">
             muc do: <strong>${meta.severity_level ?? "-"}</strong>
             | diem: <strong>${Number(meta.severity_score ?? 0).toFixed(1)}</strong>
             | goi y: ${meta.recommendation ?? "-"}
             | time: ${Number(meta.infer_time_ms ?? 0).toFixed(0)} ms
           </div>`
        : "";
    const qaInfo =
        meta && meta.qa_flag
            ? `<div style="color:#f59e0b; font-size:0.9rem; margin-bottom:0.65rem;">
                 can xem lai thu cong: ${(meta.qa?.reasons || []).join(", ")}
               </div>`
            : "";

    deepSection.innerHTML = `
        <div class="card mt-4" style="background:rgba(13,17,23,0.9); border:1px solid rgba(37,99,235,0.3);">
            <div class="card-body">
                <h4 class="mb-3"><i class="bi bi-bounding-box-circles me-2"></i>Ket qua phan tich 5 lop</h4>
                ${severityInfo}
                ${qaInfo}
                <div class="mb-3 text-center">
                    <button class="btn btn-sm btn-outline-light me-2" onclick="filterDetections('all')">Tat ca</button>
                    <button class="btn btn-sm btn-outline-light" onclick="toggleDetectionBoxes()">An/Hien khung</button>
                </div>
                <div class="row">${cards}</div>
                <div class="text-center mt-3">
                    <button class="btn btn-success me-2" onclick="downloadReport()">Tai bao cao</button>
                    <button class="btn btn-outline-primary" onclick="shareResult()">Chia se</button>
                </div>
            </div>
        </div>
    `;
}

function filterDetections(typeIndex) {
    if (!deepAnalysisData.length) return;
    if (typeIndex === "all") {
        drawDetections(deepAnalysisData, (cls) => classColor(cls), 3);
        return;
    }
    const filtered = deepAnalysisData.filter((d) => d.class_id === Number(typeIndex));
    drawDetections(filtered, (cls) => classColor(cls), 3);
}

function toggleDetectionBoxes() {
    const canvas = document.getElementById("detectionCanvas");
    canvasVisible = !canvasVisible;
    canvas.style.opacity = canvasVisible ? "1" : "0";
}

function resetAnalysis() {
    newImageToken();
    currentFile = null;
    currentImageDataUrl = null;
    currentInputSource = "upload";
    document.getElementById("fileInput").value = "";
    document.getElementById("uploadSection").style.display = "block";
    document.getElementById("analysisSection").style.display = "none";
    document.getElementById("analysisProgress").style.display = "none";
    document.getElementById("basicResultCard").style.display = "none";
    resetDeepAnalysisUI(false);
    clearCanvas();
    const streamSummarySection = document.getElementById("streamSummarySection");
    if (streamSummarySection) {
        streamSummarySection.style.display = "none";
        streamSummarySection.innerHTML = "";
    }
    if (streamSummarySessionId) {
        callJsonApi(API_STREAM_DELETE(streamSummarySessionId), "DELETE").catch(() => {});
        streamSummarySessionId = null;
    }
}

async function loadHistory(limit = 30) {
    try {
        const qs = new URLSearchParams();
        qs.set("limit", String(Number(limit) || 30));
        if (currentProjectId) qs.set("project_id", String(currentProjectId));
        const out = await callJsonApi(`${API_HISTORY}?${qs.toString()}`);
        analysisHistory = Array.isArray(out.items) ? out.items : [];
    } catch (_) {
        analysisHistory = [];
    }
}

function renderHistory() {
    const historyList = document.getElementById("historyList");
    if (!analysisHistory.length) {
        historyList.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="bi bi-inbox" style="font-size:2rem;"></i>
                <p class="mt-2">Chua co lich su</p>
            </div>
        `;
        return;
    }

    historyList.innerHTML = analysisHistory
        .map(
            (item) => {
                const hasCrack = Boolean(item.has_crack ?? item.hasCrack);
                return `
            <div class="history-item">
                <div class="history-item-header">
                    <span class="history-item-title">Phan tich #${item.id}</span>
                    <span class="history-item-status ${hasCrack ? "has-crack" : "no-crack"}">
                        ${hasCrack ? "CO VET NUT" : "KHONG"}
                    </span>
                </div>
                <div class="history-item-time"><i class="bi bi-clock"></i> ${new Date((item.ts || 0) * 1000).toLocaleString("vi-VN")}</div>
                <div style="font-size:0.85rem; color:rgba(255,255,255,0.7);">
                    Do tin cay: ${(normalizeConfidence01(item.max_confidence || 0) * 100).toFixed(1)}%
                    | box: ${Number(item.detections_count || 0)}
                    | nguon: ${item.input_source || "-"}
                </div>
                <div style="font-size:0.8rem; color:rgba(255,255,255,0.55); margin-top:0.25rem;">
                    du an: ${item.project_name || "-"}
                </div>
                <div style="margin-top:0.35rem; display:flex; gap:0.45rem; flex-wrap:wrap;">
                    ${item.input_path ? `<a class="btn btn-sm btn-outline-light" href="/api/history/${item.id}/artifact/input" target="_blank">Input</a>` : ""}
                    ${item.output_path ? `<a class="btn btn-sm btn-outline-primary" href="/api/history/${item.id}/artifact/output" target="_blank">Output</a>` : ""}
                </div>
            </div>
        `;
            }
        )
        .join("");
}

async function clearHistory() {
    if (!confirm("Ban chac chan muon xoa lich su?")) return;
    try {
        const qs = currentProjectId ? `?project_id=${currentProjectId}` : "";
        await callJsonApi(`${API_HISTORY}${qs}`, "DELETE");
    } catch (_) {}
    analysisHistory = [];
    await loadProjects().catch(() => {});
    renderHistory();
}

async function showAllHistory() {
    await loadHistory(200);
    renderHistory();
    alert(`Da tai toi da 200 ban ghi lich su${currentProjectId ? " cua du an dang chon" : ""}.`);
}

async function openCamera() {
    try {
        if (cameraStream) {
            if (cameraModal) cameraModal.show();
            return;
        }
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        const video = document.getElementById("cameraStream");
        video.srcObject = cameraStream;
        video.onloadedmetadata = () => {
            setupStreamOverlay();
            clearStreamOverlay();
        };
        setStreamButtonsState();
        setStreamLiveStatus("Camera da san sang, bam 'Bat stream realtime' de bat dau");
        if (cameraModal) cameraModal.show();
    } catch (error) {
        alert("Khong the truy cap camera.");
        console.error(error);
    }
}

async function captureImage() {
    if (streamRunning) {
        alert("Dang stream realtime. Vui long dung stream truoc khi chup anh.");
        return;
    }
    const video = document.getElementById("cameraStream");
    const canvas = document.getElementById("cameraCanvas");
    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    // Camera frames are usually noisier than uploaded files; lightly enhance contrast before inference.
    context.filter = "contrast(1.2) brightness(1.05)";
    context.drawImage(video, 0, 0);
    context.filter = "none";

    currentInputSource = "camera";
    currentImageDataUrl = canvas.toDataURL(CAMERA_CAPTURE_MIME, CAMERA_CAPTURE_JPEG_QUALITY);
    const ext = CAMERA_CAPTURE_MIME === "image/png" ? "png" : "jpg";
    currentFile = dataURLtoFile(currentImageDataUrl, `capture_${Date.now()}.${ext}`);
    const token = newImageToken();
    stopCameraStream();
    if (cameraModal) cameraModal.hide();

    showAnalysisSection(currentImageDataUrl);
    await performBasicAnalysis(token, {
        conf: CAMERA_DEEP_CONF,
        iou: CAMERA_IOU,
        imgsz: CAMERA_DEEP_IMGSZ,
    });
}

async function startRealtimeScan() {
    try {
        if (!cameraStream) {
            await openCamera();
        }
        if (!cameraStream) return;
        if (streamRunning) return;
        if (streamSummarySessionId) {
            await callJsonApi(API_STREAM_DELETE(streamSummarySessionId), "DELETE").catch(() => {});
            streamSummarySessionId = null;
        }

        const started = await callJsonApi(API_STREAM_START, "POST");
        streamSessionId = started.session_id;
        streamRunning = true;
        streamRequestBusy = false;
        streamFrameIndex = 0;
        streamProcessedFrames = 0;
        streamCandidateFrames = 0;
        streamCandidateStore = {};
        streamStartTsMs = Date.now();
        const streamSummarySection = document.getElementById("streamSummarySection");
        if (streamSummarySection) {
            streamSummarySection.style.display = "none";
            streamSummarySection.innerHTML = "";
        }

        setStreamButtonsState();
        setStreamLiveStatus(`Dang stream realtime... session=${streamSessionId.slice(0, 8)}`, "running");

        streamLoopTimer = setInterval(async () => {
            if (!streamRunning || streamRequestBusy) return;
            if (!cameraStream) return;

            const video = document.getElementById("cameraStream");
            if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;

            const activeSessionId = streamSessionId;
            if (!activeSessionId) return;

            streamRequestBusy = true;
            try {
                const canvas = document.getElementById("cameraCanvas");
                const context = canvas.getContext("2d");
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0);

                const blob = await new Promise((resolve) =>
                    canvas.toBlob(resolve, "image/jpeg", STREAM_FRAME_JPEG_QUALITY)
                );
                if (!blob) {
                    streamRequestBusy = false;
                    return;
                }
                const frameFile = new File([blob], `stream_${streamFrameIndex}.jpg`, { type: "image/jpeg" });
                const tsSec = (Date.now() - streamStartTsMs) / 1000.0;

                const fd = new FormData();
                fd.append("file", frameFile);
                fd.append("session_id", activeSessionId);
                fd.append("frame_index", String(streamFrameIndex));
                fd.append("timestamp_sec", String(tsSec));
                fd.append("scene", "default");

                const res = await fetch(API_STREAM_FRAME, { method: "POST", body: fd });
                if (!res.ok) {
                    throw new Error(`Stream frame fail: ${res.status}`);
                }
                const out = await res.json();
                if (!streamRunning || streamSessionId !== activeSessionId) {
                    return;
                }
                if (out.processed) {
                    streamProcessedFrames += 1;
                    const inferMs = Number(out.infer_time_ms || 0);
                    const maxConf = normalizeConfidence01(out.max_confidence || 0) * 100;
                    const hasCrack = !!out.has_crack;
                    const candidate = !!out.candidate_hit;
                    const trackCount = Number(out.track_count || 0);

                    if (candidate) {
                        streamCandidateFrames += 1;
                        streamCandidateStore[String(out.frame_index)] = {
                            file: frameFile,
                            confidence: normalizeConfidence01(out.max_confidence || 0),
                            timestamp_sec: Number(out.timestamp_sec || tsSec),
                        };
                        const keys = Object.keys(streamCandidateStore);
                        if (keys.length > STREAM_MAX_CANDIDATE_CACHE) {
                            keys
                                .sort((a, b) => Number(a) - Number(b))
                                .slice(0, keys.length - STREAM_MAX_CANDIDATE_CACHE)
                                .forEach((k) => delete streamCandidateStore[k]);
                        }
                    }

                    drawStreamOverlay(out.detections || [], { cached: false });
                    const status = hasCrack ? "Co vet nut" : "Khong vet nut";
                    const candidateLabel = candidate ? " | candidate" : "";
                    setStreamLiveStatus(
                        `F${out.frame_index} | ${status} | conf ${maxConf.toFixed(1)}% | ${inferMs.toFixed(0)} ms | tracks ${trackCount}${candidateLabel}`,
                        hasCrack ? "warn" : "ok"
                    );
                } else {
                    const cached = out.cached_detections || [];
                    const trackCount = Number(out.track_count || 0);
                    drawStreamOverlay(cached, { cached: true });
                    setStreamLiveStatus(
                        `F${out.frame_index} | skip (stride=${out.stride}) | tracks ${trackCount}`,
                        "running"
                    );
                }
                streamFrameIndex += 1;
            } catch (err) {
                console.error(err);
                setStreamLiveStatus(`Loi stream: ${err.message || err}`, "error");
            } finally {
                streamRequestBusy = false;
            }
        }, STREAM_SEND_INTERVAL_MS);
    } catch (err) {
        console.error(err);
        alert(err.message || "Khong the bat stream realtime.");
    }
}

async function stopRealtimeScan(showSummary = true) {
    if (streamLoopTimer) {
        clearInterval(streamLoopTimer);
        streamLoopTimer = null;
    }
    clearStreamOverlay();

    const wasRunning = streamRunning;
    streamRunning = false;
    setStreamButtonsState();

    if (!streamSessionId) return;

    const sid = streamSessionId;
    if (wasRunning && streamRequestBusy) {
        setStreamLiveStatus("Dang dung stream... cho xu ly frame cuoi", "running");
        await waitForStreamIdle(2500);
    }
    streamRequestBusy = false;

    if (!wasRunning && !showSummary) {
        try {
            await callJsonApi(API_STREAM_DELETE(sid), "DELETE");
        } catch (_) {}
        streamSessionId = null;
        return;
    }

    try {
        if (showSummary) {
            const summary = await callJsonApi(API_STREAM_SUMMARY(sid), "GET");
            renderStreamSummary(summary);
            setStreamLiveStatus("Da dung stream. Da tong hop cac doan nghi ngo.", "ok");
            streamSummarySessionId = sid;
        } else {
            setStreamLiveStatus("Da dung stream.", "");
            try {
                await callJsonApi(API_STREAM_DELETE(sid), "DELETE");
            } catch (_) {}
            streamSummarySessionId = null;
        }
    } catch (err) {
        console.error(err);
        setStreamLiveStatus("Dung stream nhung khong lay duoc summary.", "error");
        try {
            await callJsonApi(API_STREAM_DELETE(sid), "DELETE");
        } catch (_) {}
        streamSummarySessionId = null;
    } finally {
        streamSessionId = null;
    }
}

function renderStreamSummary(summary) {
    const section = document.getElementById("streamSummarySection");
    if (!section) return;

    const segments = summary?.segments || [];
    const topFrames = summary?.top_candidate_frames || [];

    const segHtml = segments.length
        ? segments
              .slice(0, 10)
              .map(
                  (s, idx) => `
            <tr>
              <td>#${idx + 1}</td>
              <td>${Number(s.start_ts).toFixed(2)}s</td>
              <td>${Number(s.end_ts).toFixed(2)}s</td>
              <td>${Number(s.duration_sec).toFixed(2)}s</td>
              <td>${(normalizeConfidence01(s.peak_conf) * 100).toFixed(1)}%</td>
              <td>${s.samples}</td>
            </tr>
        `
              )
              .join("")
        : `<tr><td colspan="6" class="text-center text-muted">Khong co doan nghi ngo nao.</td></tr>`;

    const frameButtons = topFrames.length
        ? topFrames
              .slice(0, 8)
              .map((f) => {
                  const fi = Number(f.frame_index);
                  const conf = (normalizeConfidence01(f.confidence) * 100).toFixed(1);
                  return `<button class="btn btn-sm btn-outline-primary me-2 mb-2" onclick="runDeepFromStreamFrame(${fi})">
                            Quet sau frame ${fi} (${conf}%)
                          </button>`;
              })
              .join("")
        : `<div class="text-muted">Khong co frame de de xuat quet chuyen sau.</div>`;

    section.style.display = "block";
    section.innerHTML = `
      <div class="card mt-4" style="background: rgba(13,17,23,0.92); border:1px solid rgba(37,99,235,0.35);">
        <div class="card-body">
          <h4 class="mb-3"><i class="bi bi-camera-video me-2"></i>Tong hop stream realtime</h4>
          <div style="color:rgba(255,255,255,0.85); margin-bottom:0.75rem;">
            processed: ${summary?.stats?.processed_frames ?? 0}
            | skipped: ${summary?.stats?.skipped_frames ?? 0}
            | candidate frames: ${summary?.stats?.candidate_frames ?? 0}
            | segments: ${summary?.stats?.segments ?? 0}
            | active tracks: ${summary?.stats?.active_tracks ?? 0}
            | total tracks: ${summary?.stats?.total_tracks_created ?? 0}
          </div>
          <div class="table-responsive mb-3">
            <table class="table table-dark table-sm align-middle">
              <thead>
                <tr><th>Doan</th><th>Start</th><th>End</th><th>Duration</th><th>Peak</th><th>Samples</th></tr>
              </thead>
              <tbody>${segHtml}</tbody>
            </table>
          </div>
          <div style="margin-bottom:0.6rem; color:#93c5fd; font-weight:600;">Chon frame de quet chuyen sau:</div>
          <div>${frameButtons}</div>
          <div class="mt-3 d-flex flex-wrap gap-2">
            <button class="btn btn-sm btn-success" onclick="downloadStreamReportJson()">Tai report JSON</button>
            <button class="btn btn-sm btn-outline-success" onclick="downloadStreamReportBundle()">Tai bundle ZIP</button>
            <button class="btn btn-sm btn-outline-danger" onclick="closeStreamSummarySession()">Dong phien stream</button>
          </div>
        </div>
      </div>
    `;
}

async function downloadStreamReportJson() {
    if (!streamSummarySessionId) {
        alert("Khong co phien stream da tong hop de tai report.");
        return;
    }
    try {
        const res = await callJsonApi(API_STREAM_REPORT(streamSummarySessionId), "GET");
        const report = res?.report || res || {};
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const sidShort = streamSummarySessionId.slice(0, 8);
        a.href = url;
        a.download = `stream_report_${sidShort}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert(err.message || "Khong the tai report JSON.");
    }
}

async function downloadStreamReportBundle() {
    if (!streamSummarySessionId) {
        alert("Khong co phien stream da tong hop de tai bundle.");
        return;
    }
    try {
        const endpoint = API_STREAM_REPORT_BUNDLE(streamSummarySessionId);
        const res = await fetch(endpoint, { method: "GET" });
        if (!res.ok) {
            throw new Error(`Bundle error: ${res.status}`);
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const dispo = res.headers.get("Content-Disposition") || "";
        const m = dispo.match(/filename=\"?([^\";]+)\"?/i);
        const fallback = `stream_report_bundle_${streamSummarySessionId.slice(0, 8)}_${Date.now()}.zip`;
        a.href = url;
        a.download = m?.[1] || fallback;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert(err.message || "Khong the tai bundle ZIP.");
    }
}

async function closeStreamSummarySession() {
    if (!streamSummarySessionId) return;
    const sid = streamSummarySessionId;
    try {
        await callJsonApi(API_STREAM_DELETE(sid), "DELETE");
    } catch (_) {}
    streamSummarySessionId = null;
    const section = document.getElementById("streamSummarySection");
    if (section) {
        section.style.display = "none";
        section.innerHTML = "";
    }
    setStreamLiveStatus("Da dong phien stream.", "");
}

async function runDeepFromStreamFrame(frameIndex) {
    const item = streamCandidateStore[String(frameIndex)];
    if (!item || !item.file) {
        alert("Khong tim thay frame trong bo nho tam. Vui long stream lai doan nay.");
        return;
    }
    const token = newImageToken();
    currentInputSource = "camera";
    currentFile = item.file;
    currentImageDataUrl = await fileToDataUrl(item.file);
    showAnalysisSection(currentImageDataUrl);
    await performBasicAnalysis(token, {
        conf: CAMERA_DEEP_CONF,
        iou: CAMERA_IOU,
        imgsz: CAMERA_DEEP_IMGSZ,
    });
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function stopCameraStream() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }
    clearStreamOverlay();
    setStreamButtonsState();
}

function downloadReport() {
    if (!deepAnalysisData.length) {
        alert("Chua co ket qua 5 lop de xuat bao cao.");
        return;
    }
    const payload = {
        timestamp: new Date().toISOString(),
        totalDetections: deepAnalysisData.length,
        meta: deepAnalysisMeta || {},
        detections: deepAnalysisData,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `deep-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function shareResult() {
    alert("Ban co the gui file JSON bao cao vua tai ve cho nhom ky thuat.");
}

function selectPlan(plan) {
    const planName = String(plan || "").trim().toLowerCase();
    const labels = {
        basic: "Goi Co ban",
        pro: "Goi Chuyen nghiep",
        enterprise: "Goi Doanh nghiep",
    };
    const label = labels[planName] || "goi nay";
    alert(`Ban da chon ${label}. Tinh nang thanh toan chua bat trong ban demo.`);
}
