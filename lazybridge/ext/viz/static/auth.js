// Token handling — extracted once from window.location.hash on load,
// then persisted to sessionStorage so that a manual page reload (F5)
// does not break the session.  sessionStorage is tab-scoped and is
// cleared when the tab is closed, so the token does not leak to other
// tabs or survive browser restarts.
//
// Priority: URL hash > sessionStorage.  The hash is cleared from the
// URL bar immediately after extraction (history.replaceState) so it is
// never logged by the server on subsequent navigations.

const _SS_KEY = "lb_viz_token";

let token = "";
const hash = window.location.hash || "";
const _m = hash.match(/[#&]t=([^&]+)/);
if (_m) {
  token = decodeURIComponent(_m[1]);
  // Persist for reload survivability before clearing the URL bar.
  try { sessionStorage.setItem(_SS_KEY, token); } catch (_) {}
  try {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  } catch (_) { /* replaceState may be blocked in some sandboxed frames */ }
} else {
  // Reload path: no hash present, but a previous load may have stored
  // the token in sessionStorage.
  try { token = sessionStorage.getItem(_SS_KEY) || ""; } catch (_) {}
}

export function withToken(path) {
  // EventSource does not support custom request headers, so the token
  // must travel as a query parameter for SSE connections.  The viz server
  // suppresses all access logs (log_message is a no-op), so query-string
  // exposure is not a concern here.  fetch()-based helpers (getJSON /
  // postJSON) continue to use the X-Token header instead.
  if (!token) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}t=${encodeURIComponent(token)}`;
}

export async function getJSON(path) {
  const headers = token ? { "X-Token": token } : {};
  const res = await fetch(path, { headers });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function postJSON(path, body) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["X-Token"] = token;
  const res = await fetch(path, {
    method: "POST",
    headers,
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
