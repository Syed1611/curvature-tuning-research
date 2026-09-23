# Stage-Wise Curvature Tuning

MS Research Project — Fall 2026

This branch contains the research extension developed from the NeurIPS 2025 paper:

**Curvature Tuning: Provable Training-free Model Steering From a Single Parameter**

The original Curvature Tuning results were first reproduced separately on the `main` branch.

This branch investigates whether curvature can be adapted at the level of network stages using only a small number of trainable parameters.

---

## Base Paper

**Paper:** Curvature Tuning: Provable Training-free Model Steering From a Single Parameter  
**Authors:** Leyang Hu, Matteo Gamba, Randall Balestriero  
**Conference:** NeurIPS 2025

Paper:

https://proceedings.neurips.cc/paper_files/paper/2025/hash/57cc9268e06500c7681e58829dca4a07-Abstract-Conference.html

Original implementation:

https://github.com/Leon-Leyang/curvature-tuning

---

## Motivation

The original paper introduces Curvature Tuning as a way to adapt pretrained neural networks by modifying activation-function curvature while keeping the original pretrained backbone weights frozen.

Two important variants are:

**S-CT — Single-Parameter Curvature Tuning**

One global β value is shared across the network.

```text
β → entire network
