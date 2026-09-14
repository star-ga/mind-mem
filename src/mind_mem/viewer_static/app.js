"use strict";
const form = document.getElementById("search");
const query = document.getElementById("query");
const status = document.getElementById("status");
const results = document.getElementById("results");
const graphForm = document.getElementById("graph");
const entity = document.getElementById("entity");
const depth = document.getElementById("depth");
const graphOutput = document.getElementById("graph-output");

function render(hits) {
  results.replaceChildren();
  hits.forEach((hit) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = String(hit._id || "(unnamed block)");
    const body = document.createElement("div");
    body.textContent = String(hit.Statement || hit.Summary || hit.Description || "");
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${String(hit.Status || "")} · ${String(hit._source_file || "")}`;
    item.append(title, body, meta);
    results.append(item);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = query.value.trim();
  if (!value) return;
  status.textContent = "Searching…";
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(value)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "search failed");
    render(payload.hits || []);
    status.textContent = `${payload.count || 0} admitted block(s)`;
  } catch (error) {
    results.replaceChildren();
    status.textContent = String(error.message || "search failed");
  }
});

graphForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = entity.value.trim();
  if (!value) return;
  graphOutput.textContent = "Loading…";
  try {
    const response = await fetch(`/api/graph?entity=${encodeURIComponent(value)}&depth=${encodeURIComponent(depth.value)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "graph lookup failed");
    graphOutput.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    graphOutput.textContent = String(error.message || "graph lookup failed");
  }
});
