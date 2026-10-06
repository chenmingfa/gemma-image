const $ = (id) => document.getElementById(id);

const FOLDER_KEY = "gemma-image-lib-folder";
let matchMode = "all";
let items = [];
let cloud = [];
let libraryCount = 0;
let activeId = null;
let jobRunning = false;
let searchTimer = 0;

function tokensOf(value) {
  return String(value || "").split(/[\s,，、;；]+/).filter(Boolean);
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

async function readJson(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.message || "请求没有成功");
  }
  return data;
}

function renderStatus(state) {
  const model = state.model;
  const pull = state.pull;
  const el = $("modelStatus");
  el.className = "status " + (model.ready ? "ok" : "bad");
  el.textContent = pull.running && pull.message ? pull.message : model.hint;
  libraryCount = state.counts.images;
  const pullBtn = $("pull");
  pullBtn.hidden = !(model.online && !model.ready);
  pullBtn.disabled = pull.running;
  pullBtn.textContent = pull.running ? "正在下载" : "下载模型";
  renderJob(state.job);
}

function renderJob(job) {
  jobRunning = job.running;
  const box = $("job");
  const visible = job.running || job.message;
  box.hidden = !visible;
  const pct = job.total ? Math.round((job.processed / job.total) * 100) : (job.running ? 4 : 0);
  $("bar").style.width = pct + "%";
  const detail = job.total
    ? `${job.message}（${job.processed}/${job.total}）`
    : job.message;
  $("jobText").textContent = detail;
  $("index").textContent = job.running ? "停止" : "开始识别";
}

function renderCloud() {
  const tokens = tokensOf($("q").value).map((token) => token.toLowerCase());
  $("cloud").innerHTML = cloud.map((tag) => {
    const on = tokens.includes(tag.name.toLowerCase()) ? " on" : "";
    return `<button type="button" class="chip${on}" data-tag="${esc(tag.name)}">${esc(tag.name)} ${tag.count}</button>`;
  }).join("");
}

function renderGrid() {
  const query = $("q").value.trim();
  const empty = $("empty");
  if (!items.length) {
    $("grid").innerHTML = "";
    empty.hidden = false;
    empty.textContent = libraryCount === 0 && !query
      ? "选一个文件夹，Gemma 3 会把看到的内容写成标签。"
      : "这几个标签还没有对应的图片。";
    $("count").textContent = "";
    return;
  }
  empty.hidden = true;
  $("count").textContent = items.length === 1 ? "1 张" : `${items.length} 张`;
  $("grid").innerHTML = items.map((item) => {
    const visual = item.missing
      ? `<div class="broken">文件不在了</div>`
      : `<img src="/api/images/${item.id}/thumb" alt="${esc(item.caption || item.filename)}" />`;
    const edited = item.user_edited ? `<span class="edited">改过</span>` : "";
    return `<button type="button" class="card" data-id="${item.id}">
      ${visual}
      <span class="meta">
        <strong>${esc(item.filename)}${edited}</strong>
        <span class="tags">${esc(item.tags.join("、"))}</span>
      </span>
    </button>`;
  }).join("");
}

async function runSearch() {
  const query = $("q").value.trim();
  const data = await readJson(await fetch(`/api/search?q=${encodeURIComponent(query)}&match=${matchMode}`));
  items = data.images;
  if (!query) libraryCount = data.total;
  renderCloud();
  renderGrid();
  if (data.truncated) {
    $("count").textContent = `显示前 ${data.images.length} 张，共 ${data.total} 张`;
  }
}

function toggleTag(tag) {
  const tokens = tokensOf($("q").value);
  const key = tag.toLowerCase();
  const index = tokens.findIndex((token) => token.toLowerCase() === key);
  if (index >= 0) tokens.splice(index, 1);
  else tokens.push(tag);
  $("q").value = tokens.join(" ");
  runSearch();
}

function openModal(id) {
  const item = items.find((entry) => entry.id === id);
  if (!item) return;
  activeId = id;
  $("modalTitle").textContent = item.filename;
  $("modalCaption").textContent = item.caption || "还没有说明";
  $("modalTags").value = item.tags.join("、");
  $("modalPath").textContent = item.path;
  $("modalMsg").textContent = "";
  const img = $("modalImg");
  img.alt = item.caption || item.filename;
  img.src = item.missing ? "" : `/api/images/${item.id}/file`;
  $("modal").hidden = false;
}

function closeModal() {
  $("modal").hidden = true;
  activeId = null;
}

function currentItem() {
  return items.find((entry) => entry.id === activeId);
}

async function tick() {
  try {
    const state = await readJson(await fetch("/api/status"));
    const wasRunning = jobRunning;
    renderStatus(state);
    if (wasRunning && !state.job.running) {
      await loadCloud();
      await runSearch();
    }
    setTimeout(tick, state.job.running || state.pull.running ? 800 : 4000);
  } catch (error) {
    $("modelStatus").textContent = "页面和本地服务暂时没连上";
    $("modelStatus").className = "status bad";
    setTimeout(tick, 4000);
  }
}

async function loadCloud() {
  const data = await readJson(await fetch("/api/tags"));
  cloud = data.tags;
  renderCloud();
}

$("folder").value = localStorage.getItem(FOLDER_KEY) || "";
$("folder").addEventListener("change", () => {
  localStorage.setItem(FOLDER_KEY, $("folder").value.trim());
});

$("pick").addEventListener("click", async () => {
  $("pick").disabled = true;
  try {
    const data = await readJson(await fetch("/api/pick-folder", { method: "POST" }));
    if (data.path) {
      $("folder").value = data.path;
      localStorage.setItem(FOLDER_KEY, data.path);
    }
  } catch (error) {
    $("job").hidden = false;
    $("jobText").textContent = error.message;
  } finally {
    $("pick").disabled = false;
  }
});

$("index").addEventListener("click", async () => {
  if (jobRunning) {
    await fetch("/api/index/cancel", { method: "POST" });
    return;
  }
  const folder = $("folder").value.trim();
  localStorage.setItem(FOLDER_KEY, folder);
  try {
    await readJson(await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, force: $("force").checked }),
    }));
  } catch (error) {
    $("job").hidden = false;
    $("jobText").textContent = error.message;
  }
});

$("pull").addEventListener("click", async () => {
  try {
    await readJson(await fetch("/api/model/pull", { method: "POST" }));
  } catch (error) {
    $("modelStatus").textContent = error.message;
  }
});

$("q").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, 180);
});

document.querySelector(".match").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  matchMode = button.dataset.match;
  for (const node of document.querySelectorAll(".match button")) {
    node.classList.toggle("on", node === button);
  }
  runSearch();
});

$("cloud").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tag]");
  if (button) toggleTag(button.dataset.tag);
});

$("grid").addEventListener("click", (event) => {
  const card = event.target.closest(".card");
  if (card) openModal(Number(card.dataset.id));
});

$("closeModal").addEventListener("click", closeModal);
$("modal").addEventListener("click", (event) => {
  if (event.target.id === "modal") closeModal();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModal();
  if (event.key === "/" && document.activeElement.tagName !== "INPUT") {
    event.preventDefault();
    $("q").focus();
  }
});

$("saveTags").addEventListener("click", async () => {
  if (!activeId) return;
  $("saveTags").disabled = true;
  try {
    const data = await readJson(await fetch(`/api/images/${activeId}/tags`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tags: tokensOf($("modalTags").value) }),
    }));
    const index = items.findIndex((entry) => entry.id === activeId);
    if (index >= 0) items[index] = data.image;
    $("modalMsg").textContent = "标签已保存";
    await loadCloud();
    renderGrid();
  } catch (error) {
    $("modalMsg").textContent = error.message;
  } finally {
    $("saveTags").disabled = false;
  }
});

$("resee").addEventListener("click", async () => {
  if (!activeId) return;
  if (!window.confirm("按当前画面重新写标签？")) return;
  $("resee").disabled = true;
  $("modalMsg").textContent = "正在重新看这张图";
  try {
    const data = await readJson(await fetch(`/api/images/${activeId}/recognize`, { method: "POST" }));
    const index = items.findIndex((entry) => entry.id === activeId);
    if (index >= 0) items[index] = data.image;
    $("modalCaption").textContent = data.image.caption || "";
    $("modalTags").value = data.image.tags.join("、");
    $("modalMsg").textContent = "已按当前画面更新";
    await loadCloud();
    renderGrid();
  } catch (error) {
    $("modalMsg").textContent = error.message;
  } finally {
    $("resee").disabled = false;
  }
});

$("reveal").addEventListener("click", async () => {
  if (!activeId) return;
  await fetch(`/api/images/${activeId}/reveal`, { method: "POST" });
});

$("remove").addEventListener("click", async () => {
  const item = currentItem();
  if (!item) return;
  if (!window.confirm("从库里去掉这张？文件还留在原文件夹。")) return;
  await fetch(`/api/images/${item.id}`, { method: "DELETE" });
  closeModal();
  await loadCloud();
  await runSearch();
});

loadCloud().then(runSearch).then(tick);
