let mediaRecorder = null;
let audioChunks = [];

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const textBtn = document.getElementById("textBtn");
const transcriptBox = document.getElementById("transcriptBox");
const outputBox = document.getElementById("outputBox");
const recordingStatus = document.getElementById("recordingStatus");
const manualText = document.getElementById("manualText");
const reportCards = document.getElementById("reportCards");
const summaryBanner = document.getElementById("summaryBanner");

const menuItems = document.querySelectorAll(".menu-item");
const views = document.querySelectorAll(".view");

function switchView(viewId) {
  views.forEach(view => {
    view.classList.remove("active-view");
  });

  menuItems.forEach(item => {
    item.classList.remove("active");
  });

  const selectedView = document.getElementById(viewId);
  const selectedButton = document.querySelector(`.menu-item[data-view="${viewId}"]`);

  if (selectedView) selectedView.classList.add("active-view");
  if (selectedButton) selectedButton.classList.add("active");
}

menuItems.forEach(item => {
  item.addEventListener("click", () => {
    const viewId = item.getAttribute("data-view");
    switchView(viewId);
  });
});

function safeValue(value, fallback = "Not detected") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function renderList(items, emptyText = "No items detected") {
  if (!Array.isArray(items) || items.length === 0) {
    return `<div class="value muted">${emptyText}</div>`;
  }

  return `
    <ul class="list">
      ${items.map(item => `<li>${item}</li>`).join("")}
    </ul>
  `;
}

function renderVitals(vitals = {}) {
  const sys = vitals.blood_pressure_systolic;
  const dia = vitals.blood_pressure_diastolic;
  const bp = (sys !== null && sys !== undefined && dia !== null && dia !== undefined)
    ? `${sys}/${dia}`
    : "Not detected";

  return `
    <div class="vitals-grid">
      <div class="vital-item">
        <span class="vital-label">Blood Pressure</span>
        <div class="vital-value">${bp}</div>
      </div>
      <div class="vital-item">
        <span class="vital-label">Heart Rate</span>
        <div class="vital-value">${safeValue(vitals.heart_rate)}</div>
      </div>
      <div class="vital-item">
        <span class="vital-label">Temperature</span>
        <div class="vital-value">${safeValue(vitals.temperature)}</div>
      </div>
      <div class="vital-item">
        <span class="vital-label">SpO2</span>
        <div class="vital-value">${safeValue(vitals.spo2)}</div>
      </div>
    </div>
  `;
}

function renderStructuredCards(result) {
  const clinical = result?.clinical || {};
  const quality = result?.quality || {};
  const coding = result?.coding || {};
  const meta = result?.meta || {};

  const warnings = quality.warnings || [];
  const missing = quality.missing_mandatory_fields || [];

  if (warnings.length > 0 || missing.length > 0) {
    summaryBanner.classList.remove("hidden");
    summaryBanner.innerHTML = `
      <strong>Validation Summary:</strong>
      ${warnings.length ? `Warnings: ${warnings.join(", ")}.` : ""}
      ${missing.length ? ` Missing fields: ${missing.join(", ")}.` : ""}
    `;
  } else {
    summaryBanner.classList.add("hidden");
    summaryBanner.innerHTML = "";
  }

  reportCards.innerHTML = `
    <div class="report-grid">
      <div class="info-card span-2">
        <span class="label">Reason for Visit</span>
        <div class="value ${clinical.reason_for_visit ? "" : "muted"}">${safeValue(clinical.reason_for_visit)}</div>
      </div>

      <div class="info-card">
        <span class="label">Visit Date / Time</span>
        <div class="value ${meta.visit_datetime ? "" : "muted"}">${safeValue(meta.visit_datetime)}</div>
      </div>

      <div class="info-card">
        <span class="label">Follow-up</span>
        <div class="value ${clinical.follow_up ? "" : "muted"}">${safeValue(clinical.follow_up)}</div>
      </div>

      <div class="info-card span-2">
        <span class="label">Vitals</span>
        ${renderVitals(clinical.vitals || {})}
      </div>

      <div class="info-card">
        <span class="label">Interventions</span>
        ${renderList(clinical.interventions, "No interventions detected")}
      </div>

      <div class="info-card">
        <span class="label">Critical Issues</span>
        ${renderList(clinical.critical_issues, "No critical issues detected")}
      </div>

      <div class="info-card">
        <span class="label">Normalized Problems</span>
        ${renderList(coding.problems_normalized, "No normalized problems detected")}
      </div>

      <div class="info-card">
        <span class="label">Quality Checks</span>
        ${
          warnings.length === 0 && missing.length === 0
            ? `<div class="value">No validation issues detected.</div>`
            : `
              <div class="value" style="margin-bottom:10px;"><strong>Warnings</strong></div>
              ${renderList(warnings, "No warnings")}
              <div class="value" style="margin:12px 0 10px;"><strong>Missing Mandatory Fields</strong></div>
              ${renderList(missing, "No missing fields")}
            `
        }
      </div>
    </div>
  `;
}

function setLoadingState(message) {
  transcriptBox.textContent = message;
  reportCards.innerHTML = `
    <div class="report-grid">
      <div class="info-card span-2">
        <span class="label">Processing</span>
        <div class="value">${message}</div>
      </div>
    </div>
  `;
  outputBox.textContent = "";
  summaryBanner.classList.add("hidden");
  summaryBanner.innerHTML = "";
}

function showReportView() {
  switchView("reportView");
}

function showJsonView() {
  switchView("jsonView");
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.start();
    recordingStatus.textContent = "Recording...";
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (error) {
    recordingStatus.textContent = "Microphone access failed.";
    console.error(error);
  }
}

async function stopRecording() {
  if (!mediaRecorder) return;

  mediaRecorder.stop();

  mediaRecorder.onstop = async () => {
    recordingStatus.textContent = "Processing audio...";
    startBtn.disabled = false;
    stopBtn.disabled = true;

    const blob = new Blob(audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    setLoadingState("Processing recorded audio...");
    showReportView();

    try {
      const response = await fetch("/process_audio", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (data.error) {
        transcriptBox.textContent = data.error;
        reportCards.innerHTML = `
          <div class="report-grid">
            <div class="info-card span-2">
              <span class="label">Error</span>
              <div class="value">${data.error}</div>
            </div>
          </div>
        `;
        outputBox.textContent = "";
        recordingStatus.textContent = "Error";
        return;
      }

      transcriptBox.textContent = data.transcript;
      outputBox.textContent = JSON.stringify(data.result, null, 2);
      renderStructuredCards(data.result);
      recordingStatus.textContent = "Done";
    } catch (error) {
      transcriptBox.textContent = "Audio processing failed.";
      reportCards.innerHTML = `
        <div class="report-grid">
          <div class="info-card span-2">
            <span class="label">Error</span>
            <div class="value">Audio processing failed.</div>
          </div>
        </div>
      `;
      outputBox.textContent = "";
      recordingStatus.textContent = "Error";
      console.error(error);
    }
  };
}

async function processText() {
  const text = manualText.value.trim();

  if (!text) {
    transcriptBox.textContent = "Please enter some text first.";
    return;
  }

  setLoadingState("Processing typed dictation...");
  showReportView();

  try {
    const response = await fetch("/process_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (data.error) {
      transcriptBox.textContent = data.error;
      reportCards.innerHTML = `
        <div class="report-grid">
          <div class="info-card span-2">
            <span class="label">Error</span>
            <div class="value">${data.error}</div>
          </div>
        </div>
      `;
      outputBox.textContent = "";
      return;
    }

    transcriptBox.textContent = data.transcript;
    outputBox.textContent = JSON.stringify(data.result, null, 2);
    renderStructuredCards(data.result);
  } catch (error) {
    transcriptBox.textContent = "Text processing failed.";
    reportCards.innerHTML = `
      <div class="report-grid">
        <div class="info-card span-2">
          <span class="label">Error</span>
          <div class="value">Text processing failed.</div>
        </div>
      </div>
    `;
    outputBox.textContent = "";
    console.error(error);
  }
}

if (startBtn) startBtn.addEventListener("click", startRecording);
if (stopBtn) stopBtn.addEventListener("click", stopRecording);
if (textBtn) textBtn.addEventListener("click", processText);