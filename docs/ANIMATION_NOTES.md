# Frontend Animation & Streaming Notes

This document summarizes the lightweight, Perplexity-style animation patterns added to the frontend.

## Goals
1. Provide perceived speed during streaming (answer + sources)
2. Avoid layout shift / expensive reflow
3. Keep GPU-friendly (transform + opacity only)
4. Accessible: respect `prefers-reduced-motion`

## Implemented Patterns

### 1. Answer Paragraph Reveal
Each markdown paragraph renders with class `answer-paragraph`, which uses a 500ms fade + translateY animation and a stagger (90ms * index, capped at 600ms). This simulates a fast type-in reveal without per-character DOM updates.

### 2. Skeleton / Shimmer
Answer loading state now uses skeleton blocks (`answer-skeleton-line` inside `answer-skeleton-paragraph`). Sources use a similar shimmering placeholder when no links yet. Shimmer uses a single linear-gradient background animation (`shimmerMove`).

### 3. Source Card Stagger
`WebSearchLinks` applies `source-card-anim` with per-item delay (idx * 90ms). Cards scale slightly then settle for a crisp cascade.

### 4. Sliding Tab Indicator & Scanning Bar
`StickyHeader` includes a transform-based sliding bar under active tabs and an animated scanning bar (`header-scanning-bar`) while streaming or loading.

### 5. Citation Popovers
`CitationLink` now uses an internal React popover (no direct DOM mutation) with hover/focus open, ESC / outside click close, keyboard activation via Enter/Space. Positioned relative with transform and subtle pop-in animation.

## Accessibility
- `@media (prefers-reduced-motion: reduce)` disables: paragraph reveal, source card stagger, shimmer, scanning bar.
- All popovers are hover + keyboard accessible; interactive element is a `role="button"` with `tabIndex=0`.

## Performance Considerations
- Avoided animating height; collapses use measured height + transform-free transitions.
- Limited concurrent animations (skeleton shimmer + a few staggered paragraphs/cards).
- No per-token re-render; paragraphs animate only when committed.

## Future Enhancements (Optional)
- Progressive streaming segmentation by sentence for even smoother reveal.
- Persisted source hover previews using a portal layer to avoid clipping inside scroll containers.
- Motion preference auto-detection tests.

---
Updated: ${new Date().toISOString()}