// ====================================================================
// DecodeLabs Project 4 — Image / Text Recognition (frontend logic)
// ====================================================================

let selectedFile = null;      // File object, if user uploaded
let selectedSample = null;    // sample filename, if chosen from grid
let psmModes = {};

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const sampleGrid = document.getElementById("sampleGrid");
const psmSelect = document.getElementById("psmSelect");
const sourceStatus = document.getElementById("sourceStatus");
const recognizeBtn = document.getElementById("recognizeBtn");
const statusLine = document.getElementById("statusLine");

const imagesRow = document.getElementById("imagesRow");
const imgOriginal = document.getElementById("imgOriginal");
const imgThreshold = document.getElementById("imgThreshold");
const imgAnnotated = document.getElementById("imgAnnotated");

const textResultCard = document.getElementById("textResultCard");
const gateStatus = document.getElementById("gateStatus");
const confidenceFill = document.getElementById("confidenceFill");
const confidencePct = document.getElementById("confidencePct");
const extractedText = document.getElementById("extractedText");
const wordBadges = document.getElementById("wordBadges");

// Keep the dropzone's original inner HTML so we can restore it
// (icon + instructions) whenever there's no file selected.
const dropzoneDefaultHTML = dropzone.innerHTML;

function setStatus(text, kind) {
  statusLine.textContent = `STATUS: ${text}`;
  statusLine.className = "status-line" + (kind ? " " + kind : "");
}

function updateSourceUI() {
  if (selectedFile) {
    sourceStatus.textContent = `file: ${selectedFile.name}`;
    sourceStatus.classList.add("ready");
    dropzone.classList.add("has-file");
  } else if (selectedSample) {
    sourceStatus.textContent = `sample: ${selectedSample}`;
    sourceStatus.classList.add("ready");
    dropzone.classList.remove("has-file");
    dropzone.innerHTML = dropzoneDefaultHTML;
  } else {
    sourceStatus.textContent = "no image selected";
    sourceStatus.classList.remove("ready");
    dropzone.classList.remove("has-file");
    dropzone.innerHTML = dropzoneDefaultHTML;
  }
  recognizeBtn.disabled = !(selectedFile || selectedSample);
}

// Shows a live preview of the uploaded image inside the dropzone.
function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    dropzone.innerHTML = `
      <img src="${e.target.result}" class="dropzone-preview" alt="preview">
      <span class="preview-filename">${file.name}</span>
    `;
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------
// LOAD SAMPLES + PSM MODES
// ---------------------------------------------------------------
async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    psmModes = data.psm_modes;

    sampleGrid.innerHTML = "";
    data.samples.forEach((s) => {
      const chip = document.createElement("div");
      chip.className = "sample-chip";
      chip.textContent = s.label;
      chip.dataset.file = s.file;
      chip.addEventListener("click", () => selectSample(chip, s.file));
      sampleGrid.appendChild(chip);
    });

    psmSelect.innerHTML = "";
    Object.entries(psmModes).forEach(([code, desc]) => {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = `--psm ${code} — ${desc}`;
      psmSelect.appendChild(opt);
    });

    updateSourceUI();
  } catch (err) {
    sampleGrid.innerHTML = `<p class="placeholder-text">Failed to load samples: ${err.message}</p>`;
  }
}

function selectSample(chipEl, file) {
  selectedSample = file;
  selectedFile = null;
  fileInput.value = "";
  document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("selected"));
  chipEl.classList.add("selected");
  updateSourceUI();
}

// ---------------------------------------------------------------
// UPLOAD HANDLING
// ---------------------------------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    selectedFile = fileInput.files[0];
    selectedSample = null;
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("selected"));
    showPreview(selectedFile);
    updateSourceUI();
  }
});

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    selectedSample = null;
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("selected"));
    showPreview(selectedFile);
    updateSourceUI();
  }
});

// ---------------------------------------------------------------
// RUN RECOGNITION PIPELINE
// ---------------------------------------------------------------
recognizeBtn.addEventListener("click", async () => {
  recognizeBtn.disabled = true;
  setStatus("running pipeline — ingesting, pre-processing, recognizing, validating…", "busy");
  imagesRow.style.display = "none";
  textResultCard.style.display = "none";

  const formData = new FormData();
  formData.append("psm", psmSelect.value);
  if (selectedFile) {
    formData.append("image", selectedFile);
  } else if (selectedSample) {
    formData.append("sample", selectedSample);
  }

  try {
    const res = await fetch("/api/recognize", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      setStatus(`error — ${data.error}`, "error");
      recognizeBtn.disabled = false;
      return;
    }

    renderResults(data);
    const gateMsg = data.gate_passed
      ? `pipeline complete — gate PASSED at ${data.avg_confidence}% avg confidence (${data.words_kept}/${data.total_words_detected} words kept).`
      : `pipeline complete — gate NOT MET (${data.avg_confidence}% avg confidence). Try a different PSM mode or a cleaner image.`;
    setStatus(gateMsg, data.gate_passed ? "ok" : "error");
  } catch (err) {
    setStatus(`network error — ${err.message}`, "error");
  } finally {
    recognizeBtn.disabled = false;
  }
});

function renderResults(data) {
  // Images
  imagesRow.style.display = "grid";
  imgOriginal.src = `data:image/png;base64,${data.images.original}`;
  imgThreshold.src = `data:image/png;base64,${data.images.thresholded}`;
  imgAnnotated.src = `data:image/png;base64,${data.images.annotated}`;

  // Text card
  textResultCard.style.display = "block";
  gateStatus.textContent = data.gate_passed ? "GATE: PASSED" : "GATE: FAILED";
  gateStatus.classList.toggle("ready", data.gate_passed);

  confidencePct.textContent = `${data.avg_confidence}% avg confidence`;
  requestAnimationFrame(() => {
    confidenceFill.style.width = `${Math.min(data.avg_confidence, 100)}%`;
  });
  confidenceFill.style.background = data.gate_passed ? "var(--success)" : "var(--accent)";

  extractedText.textContent = data.final_text || "(no text met the 80% confidence gate)";

  wordBadges.innerHTML = "";
  data.words.forEach((w) => {
    const badge = document.createElement("span");
    badge.className = "word-badge " + (w.kept ? "kept" : "dropped");
    badge.textContent = `${w.text} (${w.confidence}%)`;
    wordBadges.appendChild(badge);
  });
}

loadSamples();