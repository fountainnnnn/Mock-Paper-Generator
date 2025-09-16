// Backend origin (override with ?api=https://your-api.com)
const BACKEND_BASE_URL =
  new URLSearchParams(location.search).get("api") ||
  "http://localhost:8000";

// DOM
const form = document.getElementById("gen-form");
const statusAlert = document.getElementById("statusAlert");
const dlEl = document.getElementById("download");
const submitBtn = document.getElementById("submit-btn");

const progressWrap = document.getElementById("progressWrap");
const progressBar = document.getElementById("progressBar");

// year in footer
document.getElementById("year").textContent = new Date().getFullYear();

// AOS animations
if (window.AOS) {
  AOS.init({ duration: 800, once: true });
}

// Smooth scroll + focus
const getStartedBtn = document.getElementById("get-started");
if (getStartedBtn) {
  getStartedBtn.addEventListener("click", () => {
    setTimeout(() => {
      const fileInput = document.getElementById("file-input");
      if (fileInput) fileInput.focus();
    }, 450);
  });
}

let timer = null;
function startProgress() {
  progressWrap.classList.remove("d-none");
  progressBar.style.width = "2%";
  progressBar.classList.add("progress-bar-animated");
  let pct = 2;
  timer = setInterval(() => {
    pct = Math.min(90, pct + Math.random() * 6);
    progressBar.style.width = pct + "%";
  }, 250);
}
function finishProgress(success = true) {
  if (timer) clearInterval(timer);
  progressBar.classList.remove("progress-bar-animated");
  progressBar.style.width = "100%";
  progressBar.classList.toggle("bg-success", success);
  progressBar.classList.toggle("bg-danger", !success);
  setTimeout(() => {
    progressWrap.classList.add("d-none");
    progressBar.style.width = "0%";
    progressBar.classList.remove("bg-success", "bg-danger");
  }, 1200);
}

function showStatus(message, type = "info") {
  statusAlert.className = `alert alert-${type}`;
  statusAlert.textContent = message;
  statusAlert.classList.remove("d-none");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  dlEl.innerHTML = "";
  showStatus("Uploading and generating…", "info");
  submitBtn.disabled = true;
  startProgress();

  const fd = new FormData();
  const file = form.querySelector('input[type="file"]').files[0];
  if (!file) {
    finishProgress(false);
    showStatus("Please choose a file.", "warning");
    submitBtn.disabled = false;
    return;
  }
  fd.append("file", file);

  // Options
  fd.append("num_mocks", form.num_mocks.value || "1");
  fd.append(
    "difficulty",
    form.querySelector('input[name="difficulty"]:checked').value
  );

  // Default language
  fd.append("language", "en");

  // API key (optional)
  if (form.openai_api_key && form.openai_api_key.value) {
    fd.append("openai_api_key", form.openai_api_key.value);
  }

  console.log("Submitting form data:", [...fd.entries()]);

  try {
    const res = await fetch(`${BACKEND_BASE_URL}/generate`, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    // Download button
    dlEl.innerHTML = "";
    const a = document.createElement("a");
    a.href = url;
    a.textContent = "Download Mock Papers (ZIP)";
    a.className = "btn btn-outline-accent d-block my-1";
    a.download = "mockpapers.zip";
    dlEl.appendChild(a);

    showStatus("Done! Your mock papers are ready.", "success");
    finishProgress(true);
  } catch (err) {
    console.error(err);
    showStatus(`Error: ${err.message}`, "danger");
    finishProgress(false);
  } finally {
    submitBtn.disabled = false;
  }
});
