const form = document.getElementById("chat-form");
const messageInput = document.getElementById("message");
const sendButton = document.getElementById("send-button");
const clearButton = document.getElementById("clear-button");
const formStatus = document.getElementById("form-status");
const chatTranscript = document.getElementById("chat-transcript");
const welcomeState = document.getElementById("welcome-state");
const modeBadge = document.getElementById("mode-badge");
const sidebarMode = document.getElementById("sidebar-mode");
const sidebarStatus = document.getElementById("sidebar-status");
const followUpEmpty = document.getElementById("follow-up-empty");
const followUpList = document.getElementById("follow-up-list");
const sectionsEmpty = document.getElementById("sections-empty");
const sectionsList = document.getElementById("sections-list");

let pendingAssistantCard = null;

function autoResizeTextarea() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 220)}px`;
}

function scrollTranscriptToBottom() {
  chatTranscript.scrollTop = chatTranscript.scrollHeight;
}

function setStatus(message, isError = false) {
  formStatus.textContent = message;
  formStatus.style.color = isError ? "var(--danger)" : "";
  sidebarStatus.textContent = message || "Waiting for a message.";
  sidebarStatus.classList.toggle("muted", !message);
}

function setMode(mode) {
  const normalized = mode ? mode.replaceAll("_", " ") : "Idle";
  modeBadge.textContent = normalized;
  sidebarMode.textContent = normalized;
}

function ensureWelcomeState() {
  const hasMessages = chatTranscript.querySelector(".message-row");
  welcomeState.classList.toggle("hidden", Boolean(hasMessages));
}

function createMessageCard(role, label, message, extraClass = "") {
  const row = document.createElement("article");
  row.className = `message-row ${role}`.trim();

  const card = document.createElement("div");
  card.className = `message-card ${extraClass}`.trim();

  const title = document.createElement("span");
  title.className = "message-label";
  title.textContent = label;

  const body = document.createElement("p");
  body.className = "message-body";
  body.textContent = message;

  card.append(title, body);
  row.appendChild(card);
  return { row, card, body };
}

function appendMessage(role, label, message, extraClass = "") {
  const entry = createMessageCard(role, label, message, extraClass);
  chatTranscript.appendChild(entry.row);
  ensureWelcomeState();
  scrollTranscriptToBottom();
  return entry;
}

function resetInsights() {
  setMode("Idle");
  followUpList.innerHTML = "";
  followUpEmpty.classList.remove("hidden");
  sectionsList.innerHTML = "";
  sectionsEmpty.classList.remove("hidden");
}

function resetConversation() {
  pendingAssistantCard = null;
  chatTranscript.querySelectorAll(".message-row").forEach((node) => node.remove());
  ensureWelcomeState();
  resetInsights();
  setStatus("");
}

function renderFollowUps(questions) {
  followUpList.innerHTML = "";
  if (!questions.length) {
    followUpEmpty.classList.remove("hidden");
    return;
  }

  followUpEmpty.classList.add("hidden");
  questions.forEach((question) => {
    const item = document.createElement("li");
    item.textContent = question;
    followUpList.appendChild(item);
  });
}

function createSectionCard(entry) {
  const card = document.createElement("article");
  card.className = "section-card";

  const header = document.createElement("div");
  header.className = "section-card-header";

  const title = document.createElement("h3");
  title.textContent = `[${entry.law}] - Section ${entry.section} - ${entry.title}`;

  const score = document.createElement("span");
  score.className = "score-pill";
  score.textContent = `Score ${Number(entry.score).toFixed(2)}`;

  header.append(title, score);
  card.appendChild(header);

  const whyLabel = document.createElement("span");
  whyLabel.className = "block-label";
  whyLabel.textContent = "Why it applies";
  const whyText = document.createElement("p");
  whyText.textContent = entry.why_it_applies;

  const explanationLabel = document.createElement("span");
  explanationLabel.className = "block-label";
  explanationLabel.textContent = "Explanation";
  const explanationText = document.createElement("p");
  explanationText.textContent = entry.explanation;

  card.append(whyLabel, whyText, explanationLabel, explanationText);

  if (entry.original_text) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Original Law";
    const body = document.createElement("div");
    body.className = "original-text";
    body.textContent = entry.original_text;
    details.append(summary, body);
    card.appendChild(details);
  }

  return card;
}

function renderSections(sections) {
  sectionsList.innerHTML = "";
  if (!sections.length) {
    sectionsEmpty.classList.remove("hidden");
    return;
  }

  sectionsEmpty.classList.add("hidden");
  sections.forEach((entry) => sectionsList.appendChild(createSectionCard(entry)));
}

function beginAssistantThinking() {
  const pending = appendMessage("assistant", "Assistant", "Thinking", "thinking-dots");
  pendingAssistantCard = pending;
}

function finishAssistantMessage(message) {
  if (!pendingAssistantCard) {
    appendMessage("assistant", "Assistant", message);
    return;
  }

  pendingAssistantCard.card.classList.remove("thinking-dots");
  pendingAssistantCard.body.textContent = message || "No assistant response returned.";
  scrollTranscriptToBottom();
  pendingAssistantCard = null;
}

function showAssistantError(message) {
  if (!pendingAssistantCard) {
    appendMessage("system", "System", message);
    return;
  }

  pendingAssistantCard.row.classList.remove("assistant");
  pendingAssistantCard.row.classList.add("system");
  pendingAssistantCard.card.classList.remove("thinking-dots");
  pendingAssistantCard.body.textContent = message;
  pendingAssistantCard = null;
  scrollTranscriptToBottom();
}

async function submitChat(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    setStatus("Enter a message before sending.", true);
    return;
  }

  appendMessage("user", "You", message);
  messageInput.value = "";
  autoResizeTextarea();

  sendButton.disabled = true;
  clearButton.disabled = true;
  setStatus("Routing the query and preparing a grounded response...");
  setMode("Thinking");
  beginAssistantThinking();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = payload?.detail || "Request failed.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    finishAssistantMessage(payload.message || "");
    setMode(payload.mode || "unknown");
    renderFollowUps(payload.follow_up_questions || []);
    renderSections(payload.applicable_sections || []);
    setStatus("Response ready.");
  } catch (error) {
    showAssistantError(error.message || "Unable to process the request.");
    resetInsights();
    setStatus(error.message || "Unable to process the request.", true);
  } finally {
    sendButton.disabled = false;
    clearButton.disabled = false;
  }
}

form.addEventListener("submit", submitChat);

clearButton.addEventListener("click", () => {
  messageInput.value = "";
  autoResizeTextarea();
  resetConversation();
  messageInput.focus();
});

messageInput.addEventListener("input", autoResizeTextarea);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.sample;
    autoResizeTextarea();
    setStatus("Sample loaded. Press Send or edit the message.");
    messageInput.focus();
  });
});

resetConversation();
autoResizeTextarea();
