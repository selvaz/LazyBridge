// Pulse animation: a glowing circle (plus a fading trail) that
// travels from source to destination along the edge between them.
// Pure SVG — no D3 transitions on the trail because we want the
// blur filter to follow the head naturally.

const PULSE_DURATION_MS = 720;
const TRAIL_COUNT = 4;
const TRAIL_DELAY_MS = 55;

export function spawnPulse(svgGroup, src, dst, color) {
  if (!src || !dst) return;
  if (typeof src.x !== "number" || typeof dst.x !== "number") return;
  const t0 = performance.now();
  const head = svgGroup.append("circle")
    .attr("class", "pulse")
    .attr("r", 5)
    .attr("fill", color)
    .attr("filter", "url(#flare)");

  const trail = [];
  for (let i = 1; i <= TRAIL_COUNT; i++) {
    trail.push(svgGroup.append("circle")
      .attr("class", "pulse")
      .attr("r", 5 - i * 0.7)
      .attr("fill", color)
      .attr("opacity", 0.6 - i * 0.12)
      .attr("filter", "url(#glow)"));
  }

  function easeOutCubic(x) { return 1 - Math.pow(1 - x, 3); }

  function step(ts) {
    const elapsed = ts - t0;
    if (elapsed >= PULSE_DURATION_MS + TRAIL_COUNT * TRAIL_DELAY_MS) {
      head.remove();
      trail.forEach(c => c.remove());
      return;
    }
    moveCircle(head, src, dst, elapsed, 0, easeOutCubic);
    trail.forEach((c, i) => moveCircle(c, src, dst, elapsed, (i + 1) * TRAIL_DELAY_MS, easeOutCubic));
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function moveCircle(circle, src, dst, elapsed, delay, ease) {
  const e = elapsed - delay;
  if (e < 0) {
    circle.attr("cx", src.x).attr("cy", src.y).attr("opacity", 0);
    return;
  }
  const t = Math.min(1, e / PULSE_DURATION_MS);
  const k = ease(t);
  const x = src.x + (dst.x - src.x) * k;
  const y = src.y + (dst.y - src.y) * k;
  // Fade in fast, fade out at the end
  let op = 1;
  if (t > 0.85) op = (1 - t) / 0.15;
  circle.attr("cx", x).attr("cy", y).attr("opacity", op);
}
