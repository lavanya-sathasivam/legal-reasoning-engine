const state = {
  activeMatterId: null,
  matters: [],
  laws: [],
  lastSections: [],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function setView(view) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $$(".view").forEach((panel) => panel.classList.toggle("active", panel.id === `view-${view}`));
}

function appendMessage(role, text) {
  const transcript = $("#chat-transcript");
  const row = document.createElement("article");
  row.className = `message ${role}`;
  row.innerHTML = `<span>${role === "user" ? "You" : "Assistant"}</span><p></p>`;
  row.querySelector("p").textContent = text;
  transcript.appendChild(row);
  transcript.scrollTop = transcript.scrollHeight;
}

function renderSections(sections) {
  state.lastSections = sections || [];
  const container = $("#sections-list");
  if (!state.lastSections.length) {
    container.className = "section-list empty";
    container.textContent = "No confident section recommendation yet.";
    return;
  }
  container.className = "section-list";
  container.innerHTML = "";
  state.lastSections.forEach((section) => {
    const card = document.createElement("article");
    card.className = "section-card";
    card.innerHTML = `
      <div class="section-card-header">
        <strong>${section.law} ${section.section}</strong>
        <span>${Number(section.score || section.confidence || 0).toFixed(2)}</span>
      </div>
      <h3>${section.title}</h3>
      <p>${section.why_it_applies || section.explanation || ""}</p>
      <button class="text-button" type="button">Open original text</button>
    `;
    card.querySelector("button").addEventListener("click", () => renderLawPreview(section));
    container.appendChild(card);
  });
}

function renderMissing(payload) {
  const container = $("#missing-facts");
  const questions = payload.follow_up_questions || [];
  if (!questions.length) {
    container.className = "question-list empty";
    container.textContent = "No missing fact prompts for the latest answer.";
    return;
  }
  container.className = "question-list";
  container.innerHTML = questions.map((question) => `<button type="button">${question}</button>`).join("");
  $$("#missing-facts button").forEach((button) => {
    button.addEventListener("click", () => {
      $("#message").value = `${$("#message").value.trim()} ${button.textContent}`.trim();
      $("#message").focus();
    });
  });
}

function renderTrace(trace) {
  const container = $("#reasoning-trace");
  if (!trace || !trace.length) {
    container.className = "trace-list empty";
    container.textContent = "No reasoning trace available.";
    return;
  }
  container.className = "trace-list";
  container.innerHTML = "";
  trace.forEach((item) => {
    const block = document.createElement("article");
    const matched = (item.matched_elements || [])
      .filter((element) => element.satisfied)
      .map((element) => `<li>${element.label}: ${element.matched_facts.join(", ")}</li>`)
      .join("");
    block.innerHTML = `
      <strong>${item.source_id || "Section"}</strong>
      <span>Confidence ${Number(item.confidence || 0).toFixed(2)}</span>
      <ul>${matched || "<li>No satisfied elements shown.</li>"}</ul>
    `;
    container.appendChild(block);
  });
}

function renderLawPreview(section) {
  const preview = $("#law-preview");
  preview.className = "law-preview";
  preview.innerHTML = `
    <strong>${section.law} Section ${section.section}</strong>
    <h3>${section.title}</h3>
    <p>${section.original_text || "Original text unavailable."}</p>
  `;
}

async function submitChat(event) {
  event.preventDefault();
  const input = $("#message");
  const message = input.value.trim();
  if (!message) return;
  appendMessage("user", message);
  input.value = "";
  appendMessage("assistant", "Analyzing legal ingredients...");
  const pending = $$(".message.assistant").at(-1);
  try {
    const payload = await request("/api/chat/case", {
      method: "POST",
      body: JSON.stringify({ message, matter_id: state.activeMatterId }),
    });
    pending.querySelector("p").textContent = payload.message;
    renderSections(payload.applicable_sections);
    renderMissing(payload);
    renderTrace(payload.reasoning_trace);
  } catch (error) {
    pending.querySelector("p").textContent = error.message;
  }
}

async function loadMatters() {
  state.matters = await request("/api/matters");
  const html = state.matters.length
    ? state.matters.map((matter) => `<button class="matter-item" data-id="${matter.id}"><strong>${matter.title}</strong><span>${matter.description || "No description"}</span></button>`).join("")
    : `<div class="empty">No matters yet.</div>`;
  $("#matter-list").innerHTML = html;
  $("#matters-page-list").innerHTML = html;
  $$(".matter-item").forEach((item) => {
    item.addEventListener("click", () => {
      state.activeMatterId = Number(item.dataset.id);
      $("#active-matter-pill").textContent = `Matter #${state.activeMatterId}`;
      setView("chat");
    });
  });
}

async function createMatter() {
  const title = prompt("Matter title");
  if (!title) return;
  const description = prompt("Short description") || "";
  const matter = await request("/api/matters", {
    method: "POST",
    body: JSON.stringify({ title, description }),
  });
  state.activeMatterId = matter.id;
  $("#active-matter-pill").textContent = `Matter #${matter.id}`;
  await loadMatters();
}

async function loadLaws() {
  state.laws = await request("/api/laws");
  const html = state.laws.map((law) => `
    <article class="law-card">
      <strong>${law.law}</strong>
      <span>${law.sections} sections</span>
      <small>${law.status}</small>
    </article>
  `).join("");
  $("#law-status-list").innerHTML = state.laws.map((law) => `<div><span>${law.law}</span><strong>${law.sections}</strong></div>`).join("");
  $("#laws-page-list").innerHTML = html;
}

async function submitDoubt(event) {
  event.preventDefault();
  const payload = await request("/api/doubts", {
    method: "POST",
    body: JSON.stringify({
      law: $("#doubt-law").value.trim() || null,
      section: $("#doubt-section").value.trim() || null,
      question: $("#doubt-question").value.trim(),
      matter_id: state.activeMatterId,
    }),
  });
  const answer = $("#doubt-answer");
  answer.classList.remove("hidden");
  answer.innerHTML = `<h3>Answer</h3><p>${payload.answer}</p>`;
}

async function loadSettings() {
  const settings = await request("/api/settings");
  $("#setting-provider").value = settings.ai_provider || "deterministic";
  $("#setting-threshold").value = settings.confidence_threshold || 0.43;
  $("#setting-citation").value = settings.citation_style || "law-section";
  $("#setting-theme").value = settings.theme || "light";
}

async function saveSettings(event) {
  event.preventDefault();
  await request("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      settings: {
        ai_provider: $("#setting-provider").value,
        confidence_threshold: Number($("#setting-threshold").value),
        citation_style: $("#setting-citation").value,
        theme: $("#setting-theme").value,
      },
    }),
  });
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
  $$(".prompt-chip").forEach((chip) => chip.addEventListener("click", () => {
    $("#message").value = chip.dataset.prompt;
    $("#message").focus();
  }));
  $("#chat-form").addEventListener("submit", submitChat);
  $("#clear-button").addEventListener("click", () => {
    $("#message").value = "";
    $("#chat-transcript").innerHTML = "";
  });
  $("#new-matter-button").addEventListener("click", createMatter);
  $("#matter-create-inline").addEventListener("click", createMatter);
  $("#doubt-form").addEventListener("submit", submitDoubt);
  $("#settings-form").addEventListener("submit", saveSettings);
}

bindEvents();
loadMatters();
loadLaws();
loadSettings();
