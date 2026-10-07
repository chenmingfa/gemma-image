const $ = (id) => document.getElementById(id);

const FOLDER_KEY = "gemma-image-lib-folder";
const PAGE = 48;
let matchMode = "all";
let items = [];
let cloud = [];
let libraryCount = 0;
let activeId = null;
let jobRunning = false;
let searchTimer = 0;
let searchGen = 0;
let offset = 0;
let total = 0;
let loadingMore = false;
let thumbObserver = null;

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

function cardHtml(item) {
  const version = encodeURIComponent(item.indexed_at || "");
  const visual = item.missing
    ? `<div class="broken">文件不在了</div>`
    : `<img data-src="/api/images/${item.id}/thumb?v=${version}" alt="${esc(item.caption || item.filename)}" decoding="async" />`;
  const edited = item.user_edited ? `<span class="edited">改过</span>` : "";
  return `<button type="button" class="card" data-id="${item.id}">
    ${visual}
    <span class="meta">
      <strong>${esc(item.filename)}${edited}</strong>
      <span class="tags">${esc(item.tags.join("、"))}</span>
    </span>
  </button>`;
}

function observeThumbs(root) {
  if (!thumbObserver) {
    thumbObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const img = entry.target;
        if (img.dataset.src && !img.getAttribute("src")) {
          img.src = img.dataset.src;
        }
        thumbObserver.unobserve(img);
      }
    }, { rootMargin: "480px" });
  }
  for (const img of root.querySelectorAll("img[data-src]")) {
    if (!img.getAttribute("src")) thumbObserver.observe(img);
  }
}

function updateCard(item) {
  const card = document.querySelector(`.card[data-id="${item.id}"]`);
  if (!card) return;
  const strong = card.querySelector("strong");
  if (strong) {
    strong.textContent = item.filename;
    if (item.user_edited) {
      const mark = document.createElement("span");
      mark.className = "edited";
      mark.textContent = "改过";
      strong.appendChild(mark);
    }
  }
  const tags = card.querySelector(".tags");
  if (tags) tags.textContent = item.tags.join("、");
}

function paintGrid(batch, append) {
  const query = $("q").value.trim();
  const grid = $("grid");
  const empty = $("empty");
  const more = $("more");
  if (!items.length) {
    grid.innerHTML = "";
    empty.hidden = false;
    empty.textContent = libraryCount === 0 && !query
      ? "选一个文件夹，Gemma 3 会按品类、金属、主石写珠宝标签。"
      : "这几个标签还没有对应的图片。";
    $("count").textContent = "";
    more.hidden = true;
    return;
  }
  empty.hidden = true;
  if (append) {
    if (batch.length) grid.insertAdjacentHTML("beforeend", batch.map(cardHtml).join(""));
  } else {
    grid.innerHTML = items.map(cardHtml).join("");
  }
  $("count").textContent = items.length < total
    ? `已显示 ${items.length} / ${total} 张`
    : (total === 1 ? "1 张" : `${total} 张`);
  more.hidden = items.length >= total;
  observeThumbs(grid);
}

async function runSearch(options = {}) {
  const append = options.append === true;
  if (!append) searchGen += 1;
  const gen = searchGen;
  const query = $("q").value.trim();
  const start = append ? offset : 0;
  loadingMore = true;
  try {
    const data = await readJson(await fetch(
      `/api/search?q=${encodeURIComponent(query)}&match=${matchMode}&limit=${PAGE}&offset=${start}`
    ));
    if (gen !== searchGen) return;
    const batch = data.images;
    items = append ? items.concat(batch) : batch;
    offset = start + batch.length;
    total = data.total;
    if (!query) libraryCount = data.total;
    renderCloud();
    paintGrid(batch, append);
  } finally {
    if (gen === searchGen) loadingMore = false;
  }
  if (gen === searchGen) {
    const more = $("more");
    if (!more.hidden && more.getBoundingClientRect().top < window.innerHeight + 600) {
      loadMore();
    }
  }
}

function loadMore() {
  if (loadingMore || offset >= total) return;
  runSearch({ append: true });
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

$("grid").addEventListener("error", (event) => {
  const img = event.target;
  if (!img || img.tagName !== "IMG") return;
  const broken = document.createElement("div");
  broken.className = "broken";
  broken.textContent = "文件不在了";
  img.replaceWith(broken);
}, true);

new IntersectionObserver((entries) => {
  if (entries.some((entry) => entry.isIntersecting)) loadMore();
}, { rootMargin: "600px" }).observe($("more"));

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
    updateCard(data.image);
    await loadCloud();
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
    updateCard(data.image);
    await loadCloud();
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
