---
title: "Representational Geometry & Layer Sparsity Across Script Variations"
description: "Investigating how non-Latin token sequences deform manifold distances in quantized transformers, and why uniform pruning collapses cross-script alignment."
date: 2026-04-15
pubDate: 2026-04-15
tags: ["REPRESENTATION_THEORY", "SLMS"]
readingTime: "8 MIN READ"
---

When compressing small language models, standard pruning heuristics treat all intermediate layer representations symmetrically. However, inspecting activation manifolds reveals that non-Latin token distributions undergo geometric drift much earlier in network depth.

<aside class="sidenote">
<span class="sidenote-label">[Observation 01]</span>
Token sequences sampled across Devanagari and Latin scripts show an orthogonality divergence of Δ > 0.32 at Layer 4.
</aside>

By computing the singular value spectrum across residual stream activations, we can track the exact subspace collapse. Notice how the early projections maintain linear separability while deeper layers suffer from dimensional collapse.
$$
\mathcal{L}_{\text{sparsity}} = \sum_{l=1}^L \|\mathbf{W}_l\|_0 + \lambda D_{\text{KL}}(P_{\text{Devanagari}} \parallel P_{\text{Latin}})
$$
<div class="breakout-figure">
  <div style="font-family: monospace; font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
    // FIG 1.0: SINGULAR VALUE SPECTRUM (EXPANDED BREAKOUT CANVAS)
  </div>
  <div style="height: 140px; border: 1px dashed #d1d5db; display: flex; align-items: center; justify-content: center; font-family: monospace; font-size: 12px; color: #6b7280; background: #fafafa;">
    [Interactive Plot / Subspace Manifold Chart]
  </div>
</div>

This collapse demonstrates why naive magnitude pruning disproportionately degrades performance on non-Latin scripts.