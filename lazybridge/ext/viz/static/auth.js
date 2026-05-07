// Token handling — extracted once from window.location.hash on load.
// The token is sent only via the X-Token request header to avoid it
// appearing in server access logs (query-string exposure) or browser
// history (hash persistence after navigation).  The hash is cleared
// immediately after extraction so it does not survive page reload or
// appear in the browser history entry for this URL.

let token = "";
const hash = window.location.hash || "";
const _m = hash.match(/[#&]t=([^&]+)/);
if (_m) {
  token = decodeURIComponent(_m[1]);
  // Remove the token from the URL bar so it is not logged by the
  // server on the next navigation and does not persist in history.
  try {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  } catch (_) { /* replaceState may be blocked in some sandboxed frames */ }
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
