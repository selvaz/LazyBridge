// Thin wrapper over EventSource. Re-connects on drop with a tiny
// backoff and forwards every message to a single handler. The
// handler is responsible for routing by event_type.

import { withToken } from "/static/auth.js";

export function openStream(onEvent, onState) {
  let es = null;
  let backoff = 500;

  function connect() {
    es = new EventSource(withToken("/api/events"));
    es.addEventListener("open", () => {
      backoff = 500;
      onState && onState("open");
    });
    es.addEventListener("lb", (e) => {
      try {
        onEvent(JSON.parse(e.data));
      } catch (err) {
        console.error("viz: parse error", err);
      }
    });
    es.addEventListener("error", () => {
      onState && onState("retry");
      try { es.close(); } catch {}
      setTimeout(connect, Math.min(backoff, 5000));
      backoff *= 2;
    });
  }

  connect();
  return () => { if (es) try { es.close(); } catch {} };
}
