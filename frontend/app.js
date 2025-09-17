// Backend origin (override with ?api=https://your-api.com)
const BACKEND_BASE_URL =
  new URLSearchParams(location.search).get("api") ||
  "https://crystallizedcrust-mock-generator.hf.space";

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

let progressTimer = null;

function startFakeProgress() {
  let pct = 0;
  progressWrap.classList.remove("d-none");
  progressBar.classList.add("progress-bar-animated");

  progressTimer = setInterval(() => {
    if (pct < 85) {
      pct += 1; // phase 1: slow climb
    } else if (pct < 95) {
      pct += 1; // phase 2: creep
    }
    progressBar.style.width = pct + "%";
  }, 800); // 0.8s per 1% → ~68s to reach 85%, ~80s to reach 95%
}

function finishProgress(success = true) {
  if (progressTimer) clearInterval(progressTimer);
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
  showStatus(
    "Uploading your file and generating mock papers… please wait. (If it gets stuck for more than 5 minutes, refresh the page and try again)",
    "info"
  );
  submitBtn.disabled = true;

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

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${BACKEND_BASE_URL}/generate`, true);
  xhr.responseType = "blob";

  xhr.onload = () => {
    if (xhr.status === 200) {
      const blob = xhr.response;
      const url = URL.createObjectURL(blob);

      dlEl.innerHTML = "";
      const a = document.createElement("a");
      a.href = url;
      a.textContent = "Download Mock Papers (ZIP)";
      a.className = "btn btn-outline-accent d-block my-1";
      a.download = "mockpapers.zip";
      dlEl.appendChild(a);

      showStatus("Done! Your mock papers are ready.", "success");
      finishProgress(true);
    } else {
      showStatus(`Error: HTTP ${xhr.status}`, "danger");
      finishProgress(false);
    }
    submitBtn.disabled = false;
  };

  xhr.onerror = () => {
    showStatus("Network error occurred.", "danger");
    finishProgress(false);
    submitBtn.disabled = false;
  };

  xhr.send(fd);
  startFakeProgress(); // gradual 0 → 95%
});
