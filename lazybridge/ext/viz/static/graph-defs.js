// SVG <defs>: glow filter + arrow marker. Kept separate so the
// rendering module stays focused on layout/animation.

export function installDefs(svg) {
  const defs = svg.append("defs");

  // Soft outer glow used on nodes, active links, and pulses
  const glow = defs.append("filter")
    .attr("id", "glow")
    .attr("x", "-50%").attr("y", "-50%")
    .attr("width", "200%").attr("height", "200%");
  glow.append("feGaussianBlur").attr("stdDeviation", "3.5").attr("result", "blur");
  const merge = glow.append("feMerge");
  merge.append("feMergeNode").attr("in", "blur");
  merge.append("feMergeNode").attr("in", "blur");
  merge.append("feMergeNode").attr("in", "SourceGraphic");

  // Strong glow for the pulse trail
  const flare = defs.append("filter")
    .attr("id", "flare")
    .attr("x", "-100%").attr("y", "-100%")
    .attr("width", "300%").attr("height", "300%");
  flare.append("feGaussianBlur").attr("stdDeviation", "6").attr("result", "blur");
  const merge2 = flare.append("feMerge");
  merge2.append("feMergeNode").attr("in", "blur");
  merge2.append("feMergeNode").attr("in", "SourceGraphic");

  // Subtle radial gradient backing each node
  const grad = defs.append("radialGradient").attr("id", "node-grad").attr("cx", "50%").attr("cy", "40%");
  grad.append("stop").attr("offset", "0%").attr("stop-color", "rgba(40, 60, 100, 0.95)");
  grad.append("stop").attr("offset", "100%").attr("stop-color", "rgba(15, 22, 38, 0.95)");
}
