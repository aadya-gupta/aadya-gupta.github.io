---
id_tag: "EXP_07"
title: "Monosemantic Latent Geometry in Small Language Models"
category: "MECH_INTERPRETABILITY"
status: "VERIFIED"
date: 2026-04-18
stack: ["PyTorch", "SAELens", "TransformerLens", "NumPy"]
metrics:
  Feature Sparsity: "L0 ≈ 34.2"
  Cross-Entropy Recovery: "98.1%"
  Dead Latents: "< 1.2%"
repo_url: "https://github.com/aadya-gupta"
---

### Problem & Motivation
Standard transformer representations suffer from polysemanticity, where individual neurons activate for multiple unrelated semantic concepts due to superposition. We investigate whether training k-sparse autoencoders on the residual stream isolates monosemantic directions that align with biological sparse coding principles.

### Method & Probing Setup
* Trained a 16k-expansion sparse autoencoder on Layer 6 residual stream activations of a 1.5B parameter base model.
* Applied activation patching to measure causal mediation effects of identified sub-circuits during counterfactual input prompts.
* Computed cosine similarity matrices across latent directions for semantically mirrored concepts in multi-script test corpora.

### Observations & Findings
* Identified distinct, monosemantic latent vectors cleanly isolating conceptual features without collateral semantic leakage.
* Sparse autoencoder reconstruction preserved **98.1% of model task performance** while reducing effective dimensionality by an order of magnitude.
* Subspace angles between cross-lingual equivalents collapsed to near-parallel orientations ($> 0.88$ cosine similarity) once polysemantic noise was decomposed.

### Limitations
* Highly abstract conceptual features continue to exhibit minor polysemantic entanglement at lower expansion factors ($< 8\times$).