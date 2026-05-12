# Logo recolor prompt — for an image-gen LLM

The current LazyBridge logo (`lazybridge/ext/viz/static/logo.png`) uses an
electric-cyan + warm-orange palette that clashes with the proposed
Stripe-style restyle (deep purple `#635BFF` brand + amber `#FFB300` accent,
white page background with a deep-navy code surface `#0A2540`).

The goal is to **recolor** the logo while keeping the subject, pose,
composition, and overall mood. Paste the prompt below into your preferred
image model (Midjourney, DALL·E 3, Imagen, Flux, etc.) together with the
original logo as an image input / reference.

---

## Prompt — paste this verbatim

> Recolor this cartoon illustration of a young runner traversing a tech
> "circuit-board" path while floating icons (lightbulb, terminal prompt,
> gear, AI badge, isometric cube, wrench, rocket, bar chart) orbit around
> him, with the wordmark "LazyBridge" at the bottom.
>
> **Preserve exactly:**
>
> - The subject, character, pose, facial expression, and proportions.
> - The composition: runner crossing left-to-right on a curved circuit
>   path, icons distributed around him.
> - The illustration style: bold cartoon, soft cel-shading, subtle 3D
>   depth on the wordmark, glowing accents along the circuit traces.
> - The wordmark "LazyBridge" with two-tone treatment (one color for
>   "Lazy", another for "Bridge").
>
> **Change the palette to:**
>
> - **Background:** deep navy `#0A2540` (replaces the existing
>   near-black). Add a subtle radial vignette toward the corners using
>   `#14305C`.
> - **Primary glow / circuit traces / dominant accent on character
>   clothing and icons:** indigo-purple `#635BFF`. This replaces the
>   current electric cyan / bright blue everywhere it appears. The
>   circuit traces, runner's shirt, "Bridge" half of the wordmark, and
>   the dotted path should all be `#635BFF` with a softer
>   `#8B5CF6` highlight.
> - **Secondary warm accent (sparingly — rocket flame, lightbulb,
>   star on the shirt):** amber `#FFB300`. This replaces the current
>   warm orange. Use it on no more than ~15% of the surface area —
>   it should feel like the spark, not the body.
> - **"Lazy" half of the wordmark:** pure white `#FFFFFF` with a
>   subtle indigo drop shadow.
> - **"Bridge" half of the wordmark:** indigo-purple `#635BFF` with
>   a brighter `#A78BFA` top highlight for the 3D depth.
> - **Supporting icon colors:** use a muted palette derived from the
>   primary — teal-cyan `#5FCAD0` for the AI / chart icons,
>   slate `#697386` for the gear and terminal, and white-purple
>   `#EFEEFF` for the lightbulb's interior glow. Keep contrast against
>   the navy bg comfortable but not neon.
> - **Remove or desaturate:** the bright red rocket flame (replace with
>   amber `#FFB300` to `#FF8A65` gradient), any saturated bright green
>   (shift to teal `#5FCAD0`), and any pure-cyan circuit glow (shift to
>   `#635BFF`).
>
> **Atmosphere:** the new palette should feel "Stripe API docs at
> night" — sophisticated, technical, premium. Less "cyber arcade",
> more "developer toolkit with personality". Keep the playfulness of
> the character; restrain the saturation everywhere else.
>
> **Output:** 1024 × 1024 PNG with **transparent background variant**
> as well as the navy-background variant. The wordmark must remain
> crisp and legible at 200 px wide.

---

## Color cheat-sheet

| Role                          | New hex      | Replaces                  |
| ----------------------------- | ------------ | ------------------------- |
| Background (filled variant)   | `#0A2540`    | near-black                |
| Bg vignette / depth           | `#14305C`    | —                         |
| **Primary brand**             | `#635BFF`    | electric cyan / bright blue |
| Primary highlight             | `#8B5CF6`    | cyan highlight            |
| Wordmark "Lazy" half          | `#FFFFFF`    | white (kept)              |
| Wordmark "Bridge" half        | `#635BFF`    | electric blue             |
| Wordmark "Bridge" highlight   | `#A78BFA`    | —                         |
| Warm accent (spark, flame)    | `#FFB300`    | orange                    |
| Warm accent gradient stop     | `#FF8A65`    | red                       |
| Teal supporting               | `#5FCAD0`    | bright green / cyan       |
| Slate supporting              | `#697386`    | mid-grey                  |
| Lightbulb glow interior       | `#EFEEFF`    | yellow-white              |

## Two deliverable variants to request

1. **Filled** — navy `#0A2540` background, used for the docs hero,
   social-share cards, the visualizer's "Session" tab.
2. **Transparent** — same artwork, no background, used wherever the
   logo sits on a light surface (PyPI README, light-mode docs header,
   GitHub social preview when overlaid).

## Validation checklist (eyeball after generation)

- [ ] Wordmark is legible at 200 px wide.
- [ ] No pixel of pure cyan `#00B4FF` / `#1E90FF` remains.
- [ ] No pixel of saturated red remains (rocket flame is amber-orange).
- [ ] Indigo `#635BFF` is the visually dominant color, not amber.
- [ ] Background of the filled variant is `#0A2540` exactly (drop into
      a color picker to confirm — image models drift).
- [ ] Transparent variant exports cleanly without a navy halo on the
      character or wordmark edges.
- [ ] Character expression and pose match the original logo.

## If the model refuses image input (text-only fallback)

Some models can't accept an image as a strict reference. In that case,
prepend this paragraph to the prompt above:

> Imagine a square cartoon illustration: a young boy with brown hair
> running left-to-right across a curving circuit-board path, wearing
> a t-shirt with a star, blue jeans, and red shoes. Around him float
> developer-themed icons: a lightbulb, a terminal prompt window with
> `>` symbol, a gear, an "AI" badge, an isometric cube, a wrench, a
> small rocket, and a bar-chart icon. Below the scene, the wordmark
> "LazyBridge" sits in bold sans-serif with subtle 3D depth. Now
> render it with the palette below.
