const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const metaEl = document.getElementById("meta");
const formEl = document.getElementById("form");
const queryEl = document.getElementById("query");
const submitEl = document.getElementById("submit");

let ws = null;
let busy = false;
let answerText = "";

function setStatus(text, ok = true) {
  statusEl.textContent = text;
  statusEl.className = "status " + (ok ? "ok" : "err");
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/search`);

  ws.onopen = () => {
    setStatus("Connected");
    submitEl.disabled = busy || !queryEl.value.trim();
  };

  ws.onclose = () => {
    setStatus("Disconnected — reconnecting...", false);
    submitEl.disabled = true;
    busy = false;
    setTimeout(connect, 2000);
  };

  ws.onerror = () => setStatus("Connection error", false);

  ws.onmessage = (ev) => {
    try {
      handleEvent(JSON.parse(ev.data));
    } catch (e) {
      console.error(e);
    }
  };
}

function handleEvent(event) {
  const { type, data } = event;

  if (type === "text_delta") {
    answerText += data.text || "";
    outputEl.textContent = answerText;
    return;
  }

  if (type === "status" && data.message && !data.message.includes("Research started")) {
    const div = document.createElement("div");
    div.className = "tool-line";
    div.textContent = data.message;
    outputEl.appendChild(div);
    return;
  }

  if (type === "tool_use") {
    const div = document.createElement("div");
    div.className = "tool-line";
    const input = data.tool_input || {};
    const preview = input.query || input.url || input.skill || JSON.stringify(input);
    div.textContent = `Tool: ${data.tool_name} — ${preview}`;
    outputEl.appendChild(div);
    return;
  }

  if (type === "tool_result") {
    const div = document.createElement("div");
    div.className = "tool-line";
    const preview = (data.result || "").slice(0, 200);
    div.textContent = `Result (${data.tool_name}): ${preview}`;
    outputEl.appendChild(div);
    return;
  }

  if (type === "plan_update") {
    const tasks = data.tasks || [];
    const completed = data.completed_count || 0;
    const total = data.total_count || tasks.length;
    const div = document.createElement("div");
    div.className = "tool-line";
    div.textContent = `Research plan: ${completed}/${total} completed`;
    outputEl.appendChild(div);
    return;
  }

  if (type === "citation") {
    const div = document.createElement("div");
    div.className = "tool-line";
    div.textContent = `Source: ${data.title || data.url}`;
    outputEl.appendChild(div);
    return;
  }

  if (type === "error") {
    const div = document.createElement("div");
    div.className = "tool-line";
    div.textContent = `Error: ${data.message || "unknown"}`;
    outputEl.appendChild(div);
    busy = false;
    submitEl.disabled = !queryEl.value.trim();
    return;
  }

  if (type === "done") {
    if (data.final_answer) {
      outputEl.textContent = data.final_answer;
    }
    metaEl.textContent = `${data.turn_count || 0} turns · ${(data.citations || []).length} sources`;
    busy = false;
    submitEl.disabled = !queryEl.value.trim();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = queryEl.value.trim();
  if (!query || busy || !ws || ws.readyState !== WebSocket.OPEN) return;

  answerText = "";
  outputEl.textContent = "";
  metaEl.textContent = "";
  ws.send(JSON.stringify({ query }));
  busy = true;
  submitEl.disabled = true;
});

queryEl.addEventListener("input", () => {
  if (!busy) submitEl.disabled = !queryEl.value.trim();
});

connect();
