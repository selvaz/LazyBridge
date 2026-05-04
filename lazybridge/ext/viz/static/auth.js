// Token handling — extracted once from window.location.hash on load,
// then attached as a query string to every API call. The hash is
// not sent to the server in fetch() so we read it from JS and
// forward it explicitly.

let token = "";
const hash = window.location.hash || "";
const m = hash.match(/[#&]t=([^&]+)/);
if (m) token = decodeURIComponent(m[1]);

export function withToken(path) {
  if (!token) return path;
  const sep = path.includes("?") ? "&" : "?";
  return path + sep + "t=" + encodeURIComponent(token);
}

export async function getJSON(path) {
  const res = await fetch(withToken(path), { headers: { "X-Token": token } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function postJSON(path, body) {
  const res = await fetch(withToken(path), {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Token": token },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
