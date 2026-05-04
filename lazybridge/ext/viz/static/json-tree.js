// Minimal collapsible JSON renderer. Used by the inspector to
// display event payloads and store values. Plain DOM, no deps.

export function renderJSON(value, container) {
  container.innerHTML = "";
  container.appendChild(node(value, ""));
}

function node(value, key) {
  const row = document.createElement("div");
  row.className = "kv";
  if (key !== "" && key !== null && key !== undefined) {
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = key + ":";
    row.appendChild(k);
  }

  if (value === null || value === undefined) {
    return appendValue(row, "null", "null");
  }
  if (typeof value === "boolean") return appendValue(row, String(value), "b");
  if (typeof value === "number")  return appendValue(row, String(value), "n");
  if (typeof value === "string")  return appendValue(row, JSON.stringify(value), "s");

  if (Array.isArray(value)) {
    return appendCollection(row, value, "[", "]", value.length, (v, i) => node(v, String(i)));
  }
  if (typeof value === "object") {
    const keys = Object.keys(value);
    return appendCollection(row, value, "{", "}", keys.length, (_, i) => node(value[keys[i]], keys[i]));
  }
  return appendValue(row, String(value), "");
}

function appendValue(row, text, cls) {
  const v = document.createElement("span");
  v.className = "v " + cls;
  v.textContent = text;
  row.appendChild(v);
  return row;
}

function appendCollection(row, value, open, close, count, factory) {
  if (count === 0) return appendValue(row, open + close, "");
  const fold = document.createElement("span");
  fold.className = "v fold";
  fold.textContent = `${open} ${count} item${count === 1 ? "" : "s"} ${close}`;
  row.appendChild(fold);

  const nested = document.createElement("div");
  nested.className = "nested";
  for (let i = 0; i < count; i++) nested.appendChild(factory(null, i));
  row.appendChild(nested);

  let open_ = true;
  fold.addEventListener("click", (ev) => {
    ev.stopPropagation();
    open_ = !open_;
    nested.style.display = open_ ? "" : "none";
    fold.textContent = open_
      ? `${open} ${count} item${count === 1 ? "" : "s"} ${close}`
      : `${open}…${close}`;
  });
  return row;
}
